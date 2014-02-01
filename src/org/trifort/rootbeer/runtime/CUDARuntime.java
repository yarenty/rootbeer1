package org.trifort.rootbeer.runtime;

import java.io.File;
import java.util.List;

public class CUDARuntime implements IRuntime {

  private List<GpuDevice> m_cards;
  
  public CUDARuntime(){
    File native_runtime = new File("csrc/rootbeer_cuda_runtime_x64.so.1");
    System.load(native_runtime.getAbsolutePath());
    
    m_cards = loadGpuDevices();
  }

  @Override
  public List<GpuDevice> getGpuDevices() {
    return m_cards;
  }

  private native List<GpuDevice> loadGpuDevices();
}
