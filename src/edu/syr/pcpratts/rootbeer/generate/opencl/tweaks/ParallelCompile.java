/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.generate.opencl.tweaks;

import edu.syr.pcpratts.rootbeer.configuration.RootbeerPaths;
import edu.syr.pcpratts.rootbeer.runtime2.cuda.BlockingQueue;
import edu.syr.pcpratts.rootbeer.util.CompilerRunner;
import edu.syr.pcpratts.rootbeer.util.CudaPath;
import edu.syr.pcpratts.rootbeer.util.WindowsCompile;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;

public class ParallelCompile implements Runnable {

  private BlockingQueue<ParallelCompileJob> m_toCores;
  private BlockingQueue<ParallelCompileJob> m_fromCores;
  
  public ParallelCompile(){
    m_toCores = new BlockingQueue<ParallelCompileJob>();
    m_fromCores = new BlockingQueue<ParallelCompileJob>();
    
    int num_cores = Runtime.getRuntime().availableProcessors();
    if(num_cores > 2){
      num_cores = 2;
    }
    
    for(int i = 0; i < num_cores; ++i){
      Thread thread = new Thread(this);
      thread.setDaemon(true);
      thread.start();
    }
  }
  
  public CompileResult[] compile(File generated, CudaPath cuda_path, 
    String gencode_options){
    
    System.out.println("compiling CUDA code for 32bit and 64bit...");
    
    m_toCores.put(new ParallelCompileJob(generated, cuda_path, gencode_options, true));
    m_toCores.put(new ParallelCompileJob(generated, cuda_path, gencode_options, false));
    
    ParallelCompileJob ret1 = m_fromCores.take();
    ParallelCompileJob ret2 = m_fromCores.take();
    
    CompileResult[] ret = new CompileResult[2];
    if(ret1.getResult().is32Bit()){
      ret[0] = ret1.getResult();
      ret[1] = ret2.getResult();
    } else {
      ret[0] = ret2.getResult();
      ret[1] = ret1.getResult();
    }
    return ret;
  }

  public void run() {
    ParallelCompileJob job = m_toCores.take();
    job.compile();
    m_fromCores.put(job);
  }
  
}
