package org.trifort.rootbeer.runtime;

import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class Rootbeer {

  private IRuntime m_openCLRuntime;
  private IRuntime m_cudaRuntime;
  private IRuntime m_nativeRuntime;
  
  private List<StatsRow> m_stats;
  private boolean m_ranGpu;
  private ThreadConfig m_threadConfig;
  
  private List<GpuDevice> m_cards;
  
  public Rootbeer(){
    m_stats = new ArrayList<StatsRow>();
  }
  
  public List<GpuDevice> getGpuDevices(){
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
      //ignore
    }
    
    //TODO: complete the OpenCL runtime
    //try {
    //  Class c = Class.forName("org.trifort.rootbeer.runtime.OpenCLRuntime");
    //  Constructor<IRuntime> ctor = c.getConstructor();
    //  m_openCLRuntime = ctor.newInstance();
    //  m_cards.addAll(m_openCLRuntime.getGpuDevices());
    //} catch(Exception ex){
    //  //ignore
    //}
    
    return m_cards;
  }
  
  
  public void run(Kernel template, ThreadConfig thread_config){
    
  }

  public void run(List<Kernel> work, ThreadConfig thread_config) {
    
  }

  public void run(List<Kernel> work) {
    
  }
  
  public void run(Kernel template, ThreadConfig thread_config, Context context){
    context.run(template, thread_config);
  }

  public void run(List<Kernel> work, ThreadConfig thread_config, Context context) {
    
  }

  public void run(List<Kernel> work, Context context) {
    
  }
  
  /*
  public void setThreadConfig(int block_shape_x, int grid_shape_x, int numThreads){
    m_threadConfig = new ThreadConfig(block_shape_x, grid_shape_x, numThreads);
  }
  
  public void runAll(Kernel job_template){
    if(job_template instanceof CompiledKernel == false){
      System.out.println("m_ranGpu = false #1");
      m_ranGpu = false;
    }
    //this must happen above Rootbeer.runAll in case exceptions are thrown
    m_ranGpu = true;
      
    m_stats = new ArrayList<StatsRow>();
    if(m_threadConfig != null){
      m_Rootbeer.setThreadConfig(m_threadConfig);
      m_threadConfig = null;
    } else {
      m_Rootbeer.clearThreadConfig();
    }
    m_Rootbeer.runAll(job_template);
  }
  

  public void runAll(List<Kernel> jobs) {
    if(jobs.isEmpty()){
      System.out.println("m_ranGpu = false #2");
      m_ranGpu = false;
      return;
    }
    if(jobs.get(0) instanceof CompiledKernel == false){
      for(Kernel job : jobs){
        job.gpuMethod();
      }
      Kernel first = jobs.get(0);
      Class cls = first.getClass();
      Class[] ifaces = cls.getInterfaces();
      for(Class iface : ifaces){
        System.out.println("iface: "+iface.getName());
      }
      System.out.println("m_ranGpu = false 3");
      m_ranGpu = false;
    } else {
      //this must happen above Rootbeer.runAll in case exceptions are thrown
      m_ranGpu = true;
      
      m_stats = new ArrayList<StatsRow>();
      if(m_threadConfig != null){
        m_Rootbeer.setThreadConfig(m_threadConfig);
        m_threadConfig = null;
      } else {
        m_Rootbeer.clearThreadConfig();
      }
      m_Rootbeer.runAll(jobs);
    }
  }
  
  public void printMem(int start, int len){
    m_Rootbeer.printMem(start, len);
  }

    
 */ 
  
  public static void main(String[] args){
    Rootbeer rootbeer = new Rootbeer();
    List<GpuDevice> devices = rootbeer.getGpuDevices();
    System.out.println("count: "+devices.size());
    for(GpuDevice device : devices){
      System.out.println("device: "+device.getDeviceName());
      System.out.println("  id: "+device.getDeviceId());
      System.out.println("  mem: "+device.getFreeGlobalMemoryBytes());
      System.out.println("  clock: "+device.getClockRateHz());
      System.out.println("  mp_count: "+device.getMultiProcessorCount());
    }
  }
  
}
