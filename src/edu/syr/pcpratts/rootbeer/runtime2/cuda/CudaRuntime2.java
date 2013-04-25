/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.runtime2.cuda;

import jas.Pair;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicLong;

import edu.syr.pcpratts.rootbeer.configuration.Configuration;
import edu.syr.pcpratts.rootbeer.configuration.RootbeerPaths;
import edu.syr.pcpratts.rootbeer.runtime.CompiledKernel;
import edu.syr.pcpratts.rootbeer.runtime.GpuCard;
import edu.syr.pcpratts.rootbeer.runtime.Kernel;
import edu.syr.pcpratts.rootbeer.runtime.ParallelRuntime;
import edu.syr.pcpratts.rootbeer.runtime.PartiallyCompletedParallelJob;
import edu.syr.pcpratts.rootbeer.runtime.ReadOnlyAnalyzer;
import edu.syr.pcpratts.rootbeer.runtime.Rootbeer;
import edu.syr.pcpratts.rootbeer.runtime.RootbeerGpu;
import edu.syr.pcpratts.rootbeer.runtime.Serializer;
import edu.syr.pcpratts.rootbeer.runtime.StatsRow;
import edu.syr.pcpratts.rootbeer.runtime.ThreadConfig;
import edu.syr.pcpratts.rootbeer.runtime.memory.BufferPrinter;
import edu.syr.pcpratts.rootbeer.runtime.memory.Memory;
import edu.syr.pcpratts.rootbeer.runtime.util.Stopwatch;
import edu.syr.pcpratts.rootbeer.runtimegpu.GpuException;
import edu.syr.pcpratts.rootbeer.util.ResourceReader;

public class CudaRuntime2 implements ParallelRuntime {

  private static CudaRuntime2 m_Instance;

  public static CudaRuntime2 v(){
    if(m_Instance == null) {
      m_Instance = new CudaRuntime2();
    }
    return m_Instance;
  }

  private final boolean m_32bit;

  private List<GpuCard> gpuCards;
  private GpuCard currentGpuCard;

  private Handles m_Handles;
  private Handles m_ExceptionHandles;

  private int m_NumCores;

  private int m_BlockShape;
  private int m_GridShape;
  private int m_NumThreads;

  private long m_serializationTime;
  private long m_executionTime;
  private long m_deserializationTime;
  private long m_initTime;
  private long m_overallTime;

  private List<Kernel> m_JobsToWrite;
  private List<Kernel> m_JobsWritten;
  private List<Kernel> m_NotWritten;
  private List<Long> m_HandlesCache;
  private CompiledKernel m_FirstJob;
  private PartiallyCompletedParallelJob m_Partial;

  private List<Memory> m_ToSpace;
  private List<Memory> m_Texture;
  private List<ToSpaceReader> m_Readers;
  private List<ToSpaceWriter> m_Writers;

  private List<Serializer> m_serializers;

  private CpuRunner m_CpuRunner;
  private BlockShaper m_BlockShaper;

  private Stopwatch m_ctorStopwatch;
  private Stopwatch m_writeBlocksStopwatch;
  private Stopwatch m_runStopwatch;
  private Stopwatch m_runOnGpuStopwatch;
  private Stopwatch m_readBlocksStopwatch;

  private CudaRuntime2() {
    String arch = System.getProperty("os.arch");
    m_32bit = arch.equals("x86") || arch.equals("i386");

    m_ctorStopwatch = new Stopwatch();
    m_ctorStopwatch.start();
    CudaLoader loader = new CudaLoader();
    loader.load();

    m_BlockShaper = new BlockShaper();

    m_JobsToWrite = new ArrayList<Kernel>();
    m_JobsWritten = new ArrayList<Kernel>();
    m_NotWritten = new ArrayList<Kernel>();
    m_HandlesCache = new ArrayList<Long>();
    m_ToSpace = new ArrayList<Memory>();
    m_Texture = new ArrayList<Memory>();
    m_Readers = new ArrayList<ToSpaceReader>();
    m_Writers = new ArrayList<ToSpaceWriter>();

    m_writeBlocksStopwatch = new Stopwatch();
    m_runStopwatch = new Stopwatch();
    m_runOnGpuStopwatch = new Stopwatch();
    m_readBlocksStopwatch = new Stopwatch();

    // Setup gpuCard array and currentGpuCard
    gpuCards = setupGpuCards();

    // there is a bug in the concurrent serializer. setting num_cores to 1
    // right now.
    // next version of rootbeer should have a faster concurrent serializer
    // anyway
    m_NumCores = 1;
    // m_NumCores = Runtime.getRuntime().availableProcessors();

    // init GPUCard (null = currentGpuCard)
    initGpuCard(null);

    m_CpuRunner = new CpuRunner();

    m_ctorStopwatch.stop();
    m_initTime = m_ctorStopwatch.elapsedTimeMillis();

    // DEBUG
    // for(GpuCard gpuCard : gpuCards)
    // System.out.println(gpuCard.toString());

    // System.out.println("currentGpuCard: " + this.currentGpuCard.toString());

  }

  private List<GpuCard> setupGpuCards(){
    List<GpuCard> _gpuCards = null;
    File file = new File(RootbeerPaths.v().getConfigFile());

    // Check if config file has predefined gpu reserve_mem values
    if(file.exists() == false) {
      _gpuCards = setupGpuCards(m_BlockShaper.getMaxBlocksPerProc(),
          m_BlockShaper.getMaxThreadsPerBlock(), null);

      // Save generated reserve_mem values to config file
      Properties props = new Properties();
      for(GpuCard gpuCard : _gpuCards) {
        props.setProperty("gpu_" + gpuCard.getCardID() + "_reserve_mem",
            Long.toString(gpuCard.getReserveMem()));
      }
      try {
        OutputStream fout = new FileOutputStream(file);
        OutputStreamWriter writer = new OutputStreamWriter(fout);
        props.store(writer, "");
        writer.flush();
        fout.flush();
        writer.close();
        fout.close();
      } catch(Exception ex) {
        ex.printStackTrace();
      }

    } else {
      try {
        BufferedReader reader = new BufferedReader(new InputStreamReader(
            new FileInputStream(file)));
        Properties props = new Properties();
        props.load(reader);

        // Parse property file for gpu_X_reserve_mem values
        final List<Pair<Integer, Long>> reserveMemList = new ArrayList<Pair<Integer, Long>>();
        int maxGpuId = 0;
        for(int i = 0; i < props.size(); i++) {
          if(props.containsKey("gpu_" + i + "_reserve_mem")) {
            reserveMemList.add(new Pair<Integer, Long>(i, Long.parseLong(props
                .getProperty("gpu_" + i + "_reserve_mem"))));
            maxGpuId = i;
          }
        }

        // Convert List<Long> to long[]
        long[] primitiveReserveMemList = null;
        if(!reserveMemList.isEmpty()) {
          primitiveReserveMemList = new long[maxGpuId + 1];
          for(Pair<Integer, Long> p : reserveMemList) {
            Long b = p.getO2();
            primitiveReserveMemList[p.getO1()] = (b == null ? 0 : b.longValue());
          }
        }

        _gpuCards = setupGpuCards(m_BlockShaper.getMaxBlocksPerProc(),
            m_BlockShaper.getMaxThreadsPerBlock(), primitiveReserveMemList);

        reader.close();

      } catch(Exception ex) {
        ex.printStackTrace();
      }
    }
    return _gpuCards;
  }

  private void initGpuCard(GpuCard gpuCard){

    if(gpuCard != null) {
      freeCurrentGpuCard();

      this.currentGpuCard = gpuCard;
    }

    // Init native, allocate Memory...
    initCurrentGpuCard();

    m_serializers = new ArrayList<Serializer>();
    AtomicLong to_space_inst_ptr = new AtomicLong(0);
    AtomicLong to_space_static_ptr = new AtomicLong(0);
    AtomicLong texture_inst_ptr = new AtomicLong(0);
    AtomicLong texture_static_ptr = new AtomicLong(0);
    for(int i = 0; i < m_NumCores; ++i) {
      m_ToSpace.add(new FastMemory(currentGpuCard.getToSpaceAddr(),
          to_space_inst_ptr, to_space_static_ptr, currentGpuCard
              .getToSpaceSize()));
      m_Texture
          .add(new FastMemory(currentGpuCard.getTextureAddr(),
              texture_inst_ptr, texture_static_ptr, currentGpuCard
                  .getToSpaceSize()));
      m_Readers.add(new ToSpaceReader());
      m_Writers.add(new ToSpaceWriter());
    }
    m_Handles = new Handles(currentGpuCard.getHandlesAddr(),
        currentGpuCard.getGpuHandlesAddr());
    m_ExceptionHandles = new Handles(currentGpuCard.getExceptionsHandlesAddr(),
        currentGpuCard.getGpuExceptionsHandlesAddr());
  }

  public void memoryTest(){
    MemoryTest test = new MemoryTest();
    test.run(m_ToSpace.get(0));
  }

  public void run(Kernel job_template, Rootbeer rootbeer,
      ThreadConfig thread_config){
    m_runStopwatch.start();
    RootbeerGpu.setIsOnGpu(true);
    m_FirstJob = (CompiledKernel) job_template;

    if(thread_config != null) {
      m_BlockShape = thread_config.getBlockShapeX();
      m_GridShape = thread_config.getGridShapeX();
      m_NumThreads = m_BlockShape * m_GridShape;
    }
    writeSingleBlock(job_template);

    String filename;
    if(m_32bit) {
      filename = m_FirstJob.getCubin32();
    } else {
      filename = m_FirstJob.getCubin64();
    }

    if(filename.endsWith(".error")) {
      return;
    }
    if(thread_config == null) {
      calculateShape();
    }
    compileCode();

    Object gpu_thrown = null;
    try {
      runOnGpu();
      readSingleBlock(job_template);
      unload();
    } catch(Throwable ex) {
      gpu_thrown = ex;
    }

    RootbeerGpu.setIsOnGpu(false);

    m_runStopwatch.stop();
    m_overallTime = m_runStopwatch.elapsedTimeMillis();

    StatsRow stats_row = new StatsRow(m_serializationTime, m_executionTime,
        m_deserializationTime, m_overallTime, m_GridShape, m_BlockShape);

    rootbeer.addStatsRow(stats_row);
    if(gpu_thrown != null) {
      if(gpu_thrown instanceof NullPointerException) {
        NullPointerException null_ex = (NullPointerException) gpu_thrown;
        throw null_ex;
      } else if(gpu_thrown instanceof OutOfMemoryError) {
        OutOfMemoryError no_mem = (OutOfMemoryError) gpu_thrown;
        throw no_mem;
      } else if(gpu_thrown instanceof Error) {
        Error error = (Error) gpu_thrown;
        throw error;
      } else if(gpu_thrown instanceof RuntimeException) {
        RuntimeException runtime_ex = (RuntimeException) gpu_thrown;
        throw runtime_ex;
      } else {
        throw new RuntimeException("unknown exception type.");
      }
    }
  }

  public PartiallyCompletedParallelJob run(Iterator<Kernel> jobs,
      Rootbeer rootbeer, ThreadConfig thread_config){

    m_runStopwatch.start();
    RootbeerGpu.setIsOnGpu(true);
    m_Partial = new PartiallyCompletedParallelJob(jobs);

    boolean any_jobs = writeBlocks(jobs);
    if(any_jobs == false) {
      return m_Partial;
    }
    String filename = m_32bit ? m_FirstJob.getCubin32() : m_FirstJob
        .getCubin64();
    if(filename.endsWith(".error")) {
      return m_Partial;
    }
    if(thread_config == null) {
      calculateShape();
    } else {
      m_BlockShape = thread_config.getBlockShapeX();
      m_GridShape = thread_config.getGridShapeX();
      m_NumThreads = m_BlockShape * m_GridShape;
    }
    compileCode();

    Object gpu_thrown = null;
    try {
      runOnGpu();
      readBlocks();
      unload();
    } catch(Throwable ex) {
      gpu_thrown = ex;
    }

    RootbeerGpu.setIsOnGpu(false);

    m_runStopwatch.stop();
    m_overallTime = m_runStopwatch.elapsedTimeMillis();

    StatsRow stats_row = new StatsRow(m_serializationTime, m_executionTime,
        m_deserializationTime, m_overallTime, m_GridShape, m_BlockShape);

    rootbeer.addStatsRow(stats_row);
    if(gpu_thrown == null) {
      return m_Partial;
    } else {
      if(gpu_thrown instanceof NullPointerException) {
        NullPointerException null_ex = (NullPointerException) gpu_thrown;
        throw null_ex;
      } else if(gpu_thrown instanceof OutOfMemoryError) {
        OutOfMemoryError no_mem = (OutOfMemoryError) gpu_thrown;
        throw no_mem;
      } else if(gpu_thrown instanceof Error) {
        Error error = (Error) gpu_thrown;
        throw error;
      } else if(gpu_thrown instanceof RuntimeException) {
        RuntimeException runtime_ex = (RuntimeException) gpu_thrown;
        throw runtime_ex;
      } else {
        throw new RuntimeException("unknown exception type.");
      }
    }
  }

  public void writeSingleBlock(Kernel kernel){
    m_writeBlocksStopwatch.start();
    for(Memory mem : m_ToSpace) {
      mem.setAddress(0);
    }
    m_Handles.activate();
    m_Handles.resetPointer();
    m_JobsToWrite.clear();
    m_JobsWritten.clear();
    m_HandlesCache.clear();
    m_NotWritten.clear();
    m_serializers.clear();

    ReadOnlyAnalyzer analyzer = null;

    CompiledKernel compiled_kernel = (CompiledKernel) kernel;
    Memory mem = m_ToSpace.get(0);
    Memory texture_mem = m_Texture.get(0);
    mem.clearHeapEndPtr();
    texture_mem.clearHeapEndPtr();
    Serializer visitor = compiled_kernel.getSerializer(mem, texture_mem);
    visitor.setAnalyzer(analyzer);
    m_serializers.add(visitor);

    // write the statics to the heap
    m_serializers.get(0).writeStaticsToHeap();

    List<Kernel> items = new ArrayList<Kernel>();
    items.add(kernel);
    m_Writers.get(0).write(items, visitor);
    ToSpaceWriterResult result = m_Writers.get(0).join();
    List<Long> handles = result.getHandles();
    long handle = handles.get(0);
    m_HandlesCache.add(handle);
    for(int i = 0; i < m_NumThreads; ++i) {
      m_Handles.writeLong(handle);
    }

    writeClassTypeRef(m_serializers.get(0).getClassRefArray());

    m_writeBlocksStopwatch.stop();
    m_serializationTime = m_writeBlocksStopwatch.elapsedTimeMillis();

    if(Configuration.getPrintMem()) {
      BufferPrinter printer = new BufferPrinter();
      printer.print(m_ToSpace.get(0), 0, 896);
    }
  }

  public boolean writeBlocks(Iterator<Kernel> iter){
    m_writeBlocksStopwatch.start();
    for(Memory mem : m_ToSpace) {
      mem.setAddress(0);
    }
    m_Handles.activate();
    m_Handles.resetPointer();
    m_JobsToWrite.clear();
    m_JobsWritten.clear();
    m_HandlesCache.clear();
    m_NotWritten.clear();
    m_serializers.clear();

    ReadOnlyAnalyzer analyzer = null;

    boolean first_block = true;
    int count = 0;
    while(iter.hasNext()) {
      Kernel job = iter.next();
      if(first_block) {
        m_FirstJob = (CompiledKernel) job;
        first_block = false;
      }

      m_JobsToWrite.add(job);
      if(count + 1 == m_BlockShaper.getMaxThreads(currentGpuCard
          .getNumberOfMultiprocessor()))
        break;
      if(count + 1 == currentGpuCard.getNumBlocks()) {
        break;
      }
      count++;
    }
    if(count == 0) {
      return false;
    }

    for(int i = 0; i < m_NumCores; ++i) {
      Memory mem = m_ToSpace.get(i);
      Memory texture_mem = m_Texture.get(i);
      mem.clearHeapEndPtr();
      texture_mem.clearHeapEndPtr();
      Serializer visitor = m_FirstJob.getSerializer(mem, texture_mem);
      visitor.setAnalyzer(analyzer);
      m_serializers.add(visitor);
    }

    // write the statics to the heap
    m_serializers.get(0).writeStaticsToHeap();

    int items_per = m_JobsToWrite.size() / m_NumCores;
    for(int i = 0; i < m_NumCores; ++i) {
      Serializer visitor = m_serializers.get(i);
      int end_index;
      if(i == m_NumCores - 1) {
        end_index = m_JobsToWrite.size();
      } else {
        end_index = (i + 1) * items_per;
      }
      List<Kernel> items = m_JobsToWrite.subList(i * items_per, end_index);
      m_Writers.get(i).write(items, visitor);
    }

    for(int i = 0; i < m_NumCores; ++i) {
      ToSpaceWriterResult result = m_Writers.get(i).join();
      List<Long> handles = result.getHandles();
      List<Kernel> items = result.getItems();
      m_JobsWritten.addAll(items);
      m_Partial.enqueueJobs(items);
      m_HandlesCache.addAll(handles);
      m_NotWritten.addAll(result.getNotWrittenItems());
      for(Long handle : handles) {
        m_Handles.writeLong(handle);
      }
    }
    m_NumThreads = m_JobsWritten.size();
    m_Partial.addNotWritten(m_NotWritten);

    writeClassTypeRef(m_serializers.get(0).getClassRefArray());

    m_writeBlocksStopwatch.stop();
    m_serializationTime = m_writeBlocksStopwatch.elapsedTimeMillis();

    if(Configuration.getPrintMem()) {
      BufferPrinter printer = new BufferPrinter();
      printer.print(m_ToSpace.get(0), 0, 896);
    }

    return true;
  }

  private void compileCode(){
    String filename = m_32bit ? m_FirstJob.getCubin32() : m_FirstJob
        .getCubin64();
    try {
      List<byte[]> buffer = ResourceReader.getResourceArray(filename);
      int total_len = 0;
      for(byte[] sub_buffer : buffer) {
        total_len += sub_buffer.length;
      }
      loadFunction(getHeapEndPtr(), buffer, buffer.size(), total_len,
          m_NumThreads);
    } catch(Exception ex) {
      ex.printStackTrace();
    }
  }

  private void runOnGpu(){
    try {
      m_runOnGpuStopwatch.start();
      runBlocks(m_NumThreads, m_BlockShape, m_GridShape);
      m_runOnGpuStopwatch.stop();
      m_executionTime = m_runOnGpuStopwatch.elapsedTimeMillis();

    } catch(CudaErrorException ex) {
      // TODO
      // reinit(m_BlockShaper.getMaxBlocksPerProc(),
      // m_BlockShaper.getMaxThreadsPerBlock(), m_reserveMem);

      m_Handles = new Handles(currentGpuCard.getHandlesAddr(),
          currentGpuCard.getGpuHandlesAddr());
      m_ExceptionHandles = new Handles(
          currentGpuCard.getExceptionsHandlesAddr(),
          currentGpuCard.getGpuExceptionsHandlesAddr());
      throw ex;
    }
  }

  private void calculateShape(){
    m_BlockShaper.run(m_JobsWritten.size(),
        currentGpuCard.getNumberOfMultiprocessor());
    m_GridShape = m_BlockShaper.gridShape();
    m_BlockShape = m_BlockShaper.blockShape();
  }

  public void readSingleBlock(Kernel kernel){
    m_readBlocksStopwatch.start();
    m_ToSpace.get(0).setAddress(0);

    m_ExceptionHandles.activate();

    if(Configuration.getPrintMem()) {
      BufferPrinter printer = new BufferPrinter();
      printer.print(m_ToSpace.get(0), 0, 2048);
    }

    for(int i = 0; i < m_NumThreads; ++i) {
      long ref = m_ExceptionHandles.readLong();
      if(ref != 0) {
        long ref_num = ref >> 4;
        if(ref_num == m_FirstJob.getNullPointerNumber()) {
          throw new NullPointerException();
        } else if(ref_num == m_FirstJob.getOutOfMemoryNumber()) {
          throw new OutOfMemoryError();
        }
        Memory mem = m_ToSpace.get(0);
        Memory texture_mem = m_Texture.get(0);
        Serializer visitor = m_serializers.get(0);
        mem.setAddress(ref);
        Object except = visitor.readFromHeap(null, true, ref);
        if(except instanceof Error) {
          Error except_th = (Error) except;
          throw except_th;
        } else if(except instanceof GpuException) {
          GpuException gpu_except = (GpuException) except;
          System.out.println("array: " + gpu_except.m_array);
          System.out.println("index: " + gpu_except.m_arrayIndex);
          System.out.println("length: " + gpu_except.m_arrayLength);
          System.exit(1);
        } else {
          throw new RuntimeException((Throwable) except);
        }
      }
    }

    // read the statics from the heap
    m_serializers.get(0).readStaticsFromHeap();

    Serializer visitor = m_serializers.get(0);

    long handle = m_HandlesCache.get(0);
    List<Long> handles = new ArrayList<Long>();
    handles.add(handle);
    List<Kernel> jobs = new ArrayList<Kernel>();
    jobs.add(kernel);
    m_Readers.get(0).read(jobs, handles, visitor);
    m_Readers.get(0).join();

    m_readBlocksStopwatch.stop();
    m_deserializationTime = m_readBlocksStopwatch.elapsedTimeMillis();
  }

  public void readBlocks(){
    m_readBlocksStopwatch.start();
    for(int i = 0; i < m_NumCores; ++i)
      m_ToSpace.get(i).setAddress(0);

    m_ExceptionHandles.activate();

    if(Configuration.getPrintMem()) {
      BufferPrinter printer = new BufferPrinter();
      printer.print(m_ToSpace.get(0), 0, 2048);
    }

    for(int i = 0; i < m_JobsWritten.size(); ++i) {
      long ref = m_ExceptionHandles.readLong();
      if(ref != 0) {
        long ref_num = ref >> 4;
        if(ref_num == m_FirstJob.getNullPointerNumber()) {
          throw new NullPointerException();
        } else if(ref_num == m_FirstJob.getOutOfMemoryNumber()) {
          throw new OutOfMemoryError();
        }
        Memory mem = m_ToSpace.get(0);
        Memory texture_mem = m_Texture.get(0);
        Serializer visitor = m_serializers.get(0);
        mem.setAddress(ref);
        Object except = visitor.readFromHeap(null, true, ref);
        if(except instanceof Error) {
          Error except_th = (Error) except;
          throw except_th;
        } else {
          throw new RuntimeException((Throwable) except);
        }
      }
    }

    // read the statics from the heap
    m_serializers.get(0).readStaticsFromHeap();

    int items_per = m_JobsWritten.size() / m_NumCores;
    for(int i = 0; i < m_NumCores; ++i) {
      Serializer visitor = m_serializers.get(i);
      int end_index;
      if(i == m_NumCores - 1) {
        end_index = m_JobsWritten.size();
      } else {
        end_index = (i + 1) * items_per;
      }
      List<Long> handles = m_HandlesCache.subList(i * items_per, end_index);
      List<Kernel> jobs = m_JobsWritten.subList(i * items_per, end_index);
      m_Readers.get(i).read(jobs, handles, visitor);
    }

    for(int i = 0; i < m_NumCores; ++i) {
      m_Readers.get(i).join();
    }

    m_readBlocksStopwatch.stop();
    m_deserializationTime = m_readBlocksStopwatch.elapsedTimeMillis();
  }

  private long getHeapEndPtr(){
    long max = Long.MIN_VALUE;
    for(Memory mem : m_ToSpace) {
      if(mem.getHeapEndPtr() > max)
        max = mem.getHeapEndPtr();
    }
    return max;
  }

  public boolean isGpuPresent(){
    return true;
  }

  public void printMem(int start, int len){
    BufferPrinter printer = new BufferPrinter();
    printer.print(m_ToSpace.get(0), start, len);
  }

  public GpuCard getCurrentGpuCard(){
    return currentGpuCard;
  }

  public void setCurrentGpuCard(GpuCard currentGpuCard){
    initGpuCard(currentGpuCard);
  }

  public List<GpuCard> getGpuCards(){
    return gpuCards;
  }

  private native List<GpuCard> setupGpuCards(int max_blocks_per_proc,
      int max_threads_per_block, long[] reserve_mem_list);

  private native void initCurrentGpuCard();

  private native void freeCurrentGpuCard();

  private native void loadFunction(long heap_end_ptr, Object buffer, int size,
      int total_size, int num_blocks);

  private native void writeClassTypeRef(int[] refs);

  private native int runBlocks(int size, int block_shape, int grid_shape);

  private native void unload();

  public static native void printDeviceInfo();

}
