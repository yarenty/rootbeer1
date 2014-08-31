package org.trifort.rootbeer.runtime;

import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.List;


public class Rootbeer {

  private IRuntime m_cudaRuntime;
  private List<GpuDevice> m_cards;
  
  static {
    CUDALoader loader = new CUDALoader();
    loader.load();
  }
  
  public Rootbeer(){
  }
  
  public List<GpuDevice> getDevices(){
    if(m_cards != null){
      return m_cards;
    }
    
    m_cards = new ArrayList<GpuDevice>();
    try {
      Class c = Class.forName("org.trifort.rootbeer.runtime.CUDARuntime");
      Constructor<IRuntime> ctor = c.getConstructor();
      m_cudaRuntime = ctor.newInstance();
      m_cards.addAll(m_cudaRuntime.getGpuDevices());
    } catch(Exception ex){
      ex.printStackTrace();
      //ignore
    }
    
    //if(m_cards.isEmpty()){
    //  try {
    //    Class c = Class.forName("org.trifort.rootbeer.runtime.OpenCLRuntime");
    //    Constructor<IRuntime> ctor = c.getConstructor();
    //    m_openCLRuntime = ctor.newInstance();
    //    m_cards.addAll(m_openCLRuntime.getGpuDevices());
    //  } catch(Exception ex){
    //    //ignore
    //  }
    //}
    
    return m_cards;
  }
  
  public Context createDefaultContext(){
    List<GpuDevice> devices = getDevices();
    GpuDevice best = null;
    for(GpuDevice device : devices){
      if(best == null){
        best = device;
      } else {
        if(device.getMultiProcessorCount() > best.getMultiProcessorCount()){
          best = device;
        }
      }
    }
    if(best == null){
      return null;
    } else {
      return best.createContext();
    }
  }
}
