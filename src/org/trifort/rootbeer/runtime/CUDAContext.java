package org.trifort.rootbeer.runtime;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.trifort.rootbeer.configuration.Configuration;
import org.trifort.rootbeer.runtime.util.Stopwatch;
import org.trifort.rootbeer.util.ResourceReader;

import com.lmax.disruptor.EventHandler;
import com.lmax.disruptor.RingBuffer;
import com.lmax.disruptor.dsl.Disruptor;

public class CUDAContext implements Context {

  final private GpuDevice gpuDevice;
  final private boolean is32bit;

  private long nativeContext;
  private long memorySize;
  private long rootHandle;
  private byte[] cubinFile;
  private Memory objectMemory;
  private Memory textureMemory;
  private Memory exceptionsMemory;
  private Memory classMemory;
  private boolean usingKernelTemplates;
  private boolean usingUncheckedMemory;
  private long requiredMemorySize;
  private CacheConfig cacheConfig;
  private ThreadConfig threadConfig;
  private Kernel kernelTemplate;
  private CompiledKernel compiledKernel;
  
  private StatsRow stats;
  private Stopwatch writeBlocksStopwatch;
  private Stopwatch runStopwatch;
  private Stopwatch runOnGpuStopwatch;
  private Stopwatch readBlocksStopwatch;
  
  final private ExecutorService exec;
  final private Disruptor<GpuEvent> disruptor;
  final private EventHandler<GpuEvent> handler;
  final private RingBuffer<GpuEvent> ringBuffer;
  
  static {
    initializeDriver();
  }
  
  public CUDAContext(GpuDevice device){
    exec = Executors.newCachedThreadPool();
    disruptor = new Disruptor<GpuEvent>(GpuEvent.EVENT_FACTORY, 64, exec);
    handler = new GpuEventHandler();
    disruptor.handleEventsWith(handler);
    ringBuffer = disruptor.start();
    gpuDevice = device;
    memorySize = -1;
    
    String arch = System.getProperty("os.arch");
    is32bit = arch.equals("x86") || arch.equals("i386");
    
    usingUncheckedMemory = true;
    nativeContext = allocateNativeContext();
  }
  
  @Override
  public GpuDevice getDevice() {
    return gpuDevice;
  }

  @Override
  public void close() {
    disruptor.shutdown();
    exec.shutdown();
    freeNativeContext(nativeContext);
  }

  @Override
  public void setMemorySize(long memorySize) {
    this.memorySize = memorySize;
  }
  
  @Override
  public void setKernel(Kernel kernelTemplate) {
    this.kernelTemplate = kernelTemplate;
    this.compiledKernel = (CompiledKernel) kernelTemplate;
  }

  @Override
  public void setCacheConfig(CacheConfig cacheConfig) {
    this.cacheConfig = cacheConfig;
  }
  
  @Override
  public void useCheckedMemory(){
    this.usingUncheckedMemory = false;
  }
  
  @Override
  public void buildState(){
    String filename;
    int size = 0;
    boolean error;
    
    if(is32bit){
      filename = compiledKernel.getCubin32();
      size = compiledKernel.getCubin32Size();
      error = compiledKernel.getCubin32Error();
    } else {
      filename = compiledKernel.getCubin64();
      size = compiledKernel.getCubin64Size();
      error = compiledKernel.getCubin32Error();
    }

    if(error){
      throw new RuntimeException("CUDA code compiled with error");
    }
    
    cubinFile = readCubinFile(filename, size);
    
    if(usingUncheckedMemory){
      classMemory = new FixedMemory(1024);
      exceptionsMemory = new FixedMemory(getExceptionsMemSize(threadConfig));
    } else {
      exceptionsMemory = new CheckedFixedMemory(getExceptionsMemSize(threadConfig));
      classMemory = new CheckedFixedMemory(1024);
    }
    if(memorySize == -1){
      findMemorySize(cubinFile.length);
    }
    if(usingUncheckedMemory){
      objectMemory = new FixedMemory(memorySize);
    } else {
      objectMemory = new CheckedFixedMemory(memorySize);
    }
    
    long seq = ringBuffer.next();
    GpuEvent gpuEvent = ringBuffer.get(seq);
    gpuEvent.setValue(GpuEventCommand.NATIVE_BUILD_STATE);
    gpuEvent.getFuture().reset();
    ringBuffer.publish(seq);
    gpuEvent.getFuture().take();
  }
  
  private long getExceptionsMemSize(ThreadConfig thread_config) {
    if(Configuration.runtimeInstance().getExceptions()){
      return 4L*thread_config.getNumThreads();
    } else {
      return 4;
    }
  }
  
  private byte[] readCubinFile(String filename, int length) {
    try {
      byte[] buffer = ResourceReader.getResourceArray(filename, length);
      return buffer;
    } catch(Exception ex){
      throw new RuntimeException(ex);
    }
  }
  
  private void findMemorySize(int cubinFileLength){
    long freeMemSizeGPU = gpuDevice.getFreeGlobalMemoryBytes();
    long freeMemSizeCPU = Runtime.getRuntime().freeMemory();
    long freeMemSize = Math.min(freeMemSizeGPU, freeMemSizeCPU);
    
    freeMemSize -= cubinFileLength;
    freeMemSize -= exceptionsMemory.getSize();
    freeMemSize -= classMemory.getSize();
    freeMemSize -= 2048;
    
    if(freeMemSize <= 0){
      StringBuilder error = new StringBuilder();
      error.append("OutOfMemory while allocating Java CPU and GPU memory.\n");
      error.append("  Try increasing the max Java Heap Size using -Xmx and the initial Java Heap Size using -Xms.\n");
      error.append("  Try reducing the number of threads you are using.\n");
      error.append("  Try using kernel templates.\n");
      error.append("  Debugging Output:\n");
      error.append("    GPU_SIZE: "+freeMemSizeGPU+"\n");
      error.append("    CPU_SIZE: "+freeMemSizeCPU+"\n");
      error.append("    EXCEPTIONS_SIZE: "+exceptionsMemory.getSize()+"\n");
      error.append("    CLASS_MEMORY_SIZE: "+classMemory.getSize());
      throw new RuntimeException(error.toString());
    }
    memorySize = freeMemSize;
  }

  @Override
  public long getRequiredMemory() {
    // TODO Auto-generated method stub
    return 0;
  }

  @Override
  public void setThreadConfig(ThreadConfig threadConfig){
    this.threadConfig = threadConfig;
  }
  
  @Override
  public void run(){
    GpuFuture future = runAsync();
    future.take();
  }
  
  @Override
  public GpuFuture runAsync() {
    return null;
  }

  @Override
  public StatsRow getStats() {
    return stats;
  }
  
  private class GpuEventHandler implements EventHandler<GpuEvent>{
    @Override
    public void onEvent(final GpuEvent gpuEvent, final long sequence, final boolean endOfBatch)
        throws Exception {
      
      switch(gpuEvent.getValue()){
      case NATIVE_BUILD_STATE:
        boolean usingExceptions = Configuration.runtimeInstance().getExceptions();
        nativeBuildState(nativeContext, gpuDevice.getDeviceId(), cubinFile, 
            cubinFile.length, threadConfig.getBlockShapeX(), 
            threadConfig.getGridShapeX(), threadConfig.getNumThreads(), 
            objectMemory, exceptionsMemory, classMemory, 
            b2i(usingExceptions), cacheConfig.ordinal());
        gpuEvent.getFuture().signal();
        break;
      case NATIVE_RUN:
        cudaRun(nativeContext, 0);
        gpuEvent.getFuture().signal();
        break;
      }
    }
  }
  
  private int b2i(boolean value){
    if(value){
      return 1;
    } else {
      return 0;
    }
  }
  
  private static native void initializeDriver();
  private native long allocateNativeContext();
  private native void freeNativeContext(long nativeContext);
  
  private native void nativeBuildState(long nativeContext, int deviceIndex, byte[] cubinFile, 
      int cubinLength, int blockShapeX, int gridShapeX, int numThreads, 
      Memory objectMem, Memory exceptionsMem, Memory classMem, int usingExceptions, 
      int cacheConfig);
  
  private native void cudaRun(long nativeContext, int handle);
}
