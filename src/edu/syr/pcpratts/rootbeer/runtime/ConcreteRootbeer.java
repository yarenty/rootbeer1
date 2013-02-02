/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.runtime;

import edu.syr.pcpratts.rootbeer.configuration.Configuration;
import edu.syr.pcpratts.rootbeer.generate.opencl.tweaks.CudaTweaks;
import edu.syr.pcpratts.rootbeer.generate.opencl.tweaks.NativeCpuTweaks;
import edu.syr.pcpratts.rootbeer.generate.opencl.tweaks.Tweaks;
import edu.syr.pcpratts.rootbeer.runtime.cpu.CpuRuntime;
import edu.syr.pcpratts.rootbeer.runtime.nativecpu.NativeCpuRuntime;
import edu.syr.pcpratts.rootbeer.runtime2.cuda.CudaRuntime2;
import java.util.Iterator;
import java.util.List;

public class ConcreteRootbeer implements IRootbeerInternal {

  private boolean m_GpuWorking;
  private Rootbeer m_rootbeer;
  private ThreadConfig m_threadConfig;

  public ConcreteRootbeer(Rootbeer rootbeer){
    m_rootbeer = rootbeer;
    m_GpuWorking = true;
    if(Configuration.runtimeInstance().getMode() == Configuration.MODE_GPU) {
      CudaRuntime2.v();
    }
  }

  public void runAll(List<Kernel> list){
    Iterator<Kernel> iter = run(list.iterator());
    while(iter.hasNext()){
      iter.next();
    }
  }
  
  public Iterator<Kernel> run(Iterator<Kernel> iter) {
    if(Configuration.runtimeInstance().getMode() == Configuration.MODE_NEMU){
      return runOnNativeCpu(iter);
    } else if(Configuration.runtimeInstance().getMode() == Configuration.MODE_JEMU){
      return runOnCpu(iter);
    } else {
      return runOnCudaGpu(iter);
    }
  }

  private Iterator<Kernel> runOnCpu(Iterator<Kernel> jobs) {
    try {
      PartiallyCompletedParallelJob partial = CpuRuntime.v().run(jobs, m_rootbeer, m_threadConfig);
      return new ResultIterator(partial, CpuRuntime.v(), m_rootbeer);
    } catch (Exception ex){
      ex.printStackTrace();
      System.exit(-1);
      return null;
    }
  }
  
  private Iterator<Kernel> runOnCudaGpu(Iterator<Kernel> jobs) {    
    Tweaks.setInstance(new CudaTweaks());
    PartiallyCompletedParallelJob partial = CudaRuntime2.v().run(jobs, m_rootbeer, m_threadConfig);
    return new ResultIterator(partial, CudaRuntime2.v(), m_rootbeer);
  }
  
  private Iterator<Kernel> runOnNativeCpu(Iterator<Kernel> jobs) {
    Tweaks.setInstance(new NativeCpuTweaks());
    PartiallyCompletedParallelJob partial = NativeCpuRuntime.v().run(jobs, m_rootbeer, m_threadConfig);
    return new ResultIterator(partial, NativeCpuRuntime.v(), m_rootbeer);
  }

  public void setThreadConfig(ThreadConfig thread_config) {
    m_threadConfig = thread_config;
  }

  public void clearThreadConfig() {
    m_threadConfig = null;
  }
}
