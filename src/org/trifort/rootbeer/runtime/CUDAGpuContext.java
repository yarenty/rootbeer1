package org.trifort.rootbeer.runtime;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.trifort.rootbeer.configuration.Configuration;
import org.trifort.rootbeer.runtime.memory.BufferPrinter;
import org.trifort.rootbeer.runtimegpu.GpuException;
import org.trifort.rootbeer.util.ResourceReader;

public class CUDAGpuContext implements GpuContext {

  private GpuDevice m_device;
  private List<StatsRow> m_stats;
  private boolean m_32bit;
  private Map<String, byte[]> m_cubinFiles;
  
  private GpuMemory m_objectMemory;
  private GpuMemory m_handlesMemory;
  private GpuMemory m_exceptionsMemory;
  
  public CUDAGpuContext(GpuDevice device){
    m_device = device;    
    
    String arch = System.getProperty("os.arch");
    m_32bit = arch.equals("x86") || arch.equals("i386");
    
    m_cubinFiles = new HashMap<String, byte[]>();
  }

  @Override
  public void init() {
    m_objectMemory = new CUDAMemory(1000);
    m_handlesMemory = new CUDAMemory(1000);
    m_exceptionsMemory = new CUDAMemory(1000);
  }

  @Override
  public void close() {
  }
  
  public List<StatsRow> getStats(){
    return m_stats;
  }
  
  @Override
  public GpuDevice getDevice() {
    return m_device;
  }

  @Override
  public void run(Kernel template, ThreadConfig thread_config) {
    CompiledKernel compiled_kernel = (CompiledKernel) template;
    
    String filename;
    if(m_32bit){
      filename = compiled_kernel.getCubin32();
    } else {
      filename = compiled_kernel.getCubin64();
    }
    
    if(filename.endsWith(".error")){
      throw new RuntimeException("CUDA code compiled with error");
    }
    
    byte[] cubin_file;
    
    if(m_cubinFiles.containsKey(filename)){
      cubin_file = m_cubinFiles.get(filename);
    } else {
      cubin_file = readCubinFile(filename);
      m_cubinFiles.put(filename, cubin_file);
    }
    
    writeBlocksTemplate(compiled_kernel);
    runBlocks(thread_config);
    readBlocksTemplate(compiled_kernel, thread_config);
  }
  
  private void runBlocks(ThreadConfig thread_config){    
    cudaRun(thread_config.getBlockShapeX(), thread_config.getGridShapeX(), 
      thread_config.getNumThreads(), m_objectMemory, m_handlesMemory);
  }  
  
  private void writeBlocksTemplate(CompiledKernel compiled_kernel){
    m_objectMemory.reset();
    m_handlesMemory.reset();
    
    Serializer serializer = compiled_kernel.getSerializer(m_objectMemory);
    serializer.writeStaticsToHeap();
    long handle = serializer.writeToHeap(compiled_kernel);
    m_handlesMemory.writeRef(handle);
    
    if(Configuration.getPrintMem()){
      BufferPrinter printer = new BufferPrinter();
      printer.print(m_objectMemory, 0, 896);
    }
  }
  
  private void readBlocksTemplate(CompiledKernel compiled_kernel, ThreadConfig thread_config){
    m_handlesMemory.setAddress(0);
    m_exceptionsMemory.setAddress(0);
    
    Serializer serializer = compiled_kernel.getSerializer(m_objectMemory);
    
    for(int i = 0; i < thread_config.getNumThreads(); ++i){
      long ref = m_exceptionsMemory.readLong();
      if(ref != 0){
        long ref_num = ref >> 4;
        if(ref_num == compiled_kernel.getNullPointerNumber()){
          throw new NullPointerException(); 
        } else if(ref_num == compiled_kernel.getOutOfMemoryNumber()){
          throw new OutOfMemoryError();
        }
        
        m_objectMemory.setAddress(ref);           
        Object except = serializer.readFromHeap(null, true, ref);
        if(except instanceof Error){
          Error except_th = (Error) except;
          throw except_th;
        } else if(except instanceof GpuException){
          GpuException gpu_except = (GpuException) except;
          gpu_except.throwArrayOutOfBounds();
        } else {
          throw new RuntimeException((Throwable) except);
        }
      }
    }    
    
    serializer.readStaticsFromHeap();
    serializer.readFromHeap(compiled_kernel, true, m_handlesMemory.readRef());
  }

  private byte[] readCubinFile(String filename) {
    try {
      List<byte[]> buffer = ResourceReader.getResourceArray(filename);
      int total_len = 0;
      for(byte[] sub_buffer : buffer){
        total_len += sub_buffer.length;
      }
      byte[] cubin_file = new byte[total_len];
      int pos = 0;
      for(byte[] small_buffer : buffer){
        System.arraycopy(small_buffer, 0, cubin_file, pos, small_buffer.length);
        pos += small_buffer.length;
      }
      return cubin_file;
    } catch(Exception ex){
      throw new RuntimeException(ex);
    }
  }
  
  @Override
  public void run(List<Kernel> work, ThreadConfig thread_config) {
    
  }
  
  private native void cudaRun(int block_shape_x, int grid_shape_x, 
    int num_threads, GpuMemory object_mem, GpuMemory handles_mem);
}
