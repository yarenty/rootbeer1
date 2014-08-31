package org.trifort.rootbeer.runtime;

import java.io.Closeable;
import java.util.List;

public interface Context extends Closeable {

  public GpuDevice getDevice();
  public void setMemorySize(long memorySize);
  public void setCacheConfig(CacheConfig config);
  public void setThreadConfig(ThreadConfig thread_config);
  public void setThreadConfig(int block_shape_x, int grid_shape_x, int num_threads);
  public void setKernel(Kernel kernelTemple);
  public void useCheckedMemory();
  public void buildState();
  public void run();
  public GpuFuture runAsync();
  public long getRequiredMemory();
  public void close();
  public StatsRow getStats();
  
}
