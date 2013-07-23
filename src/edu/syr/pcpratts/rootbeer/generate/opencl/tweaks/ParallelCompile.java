/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.generate.opencl.tweaks;

import java.io.File;
import java.util.LinkedList;
import java.util.List;

import edu.syr.pcpratts.rootbeer.generate.opencl.tweaks.GencodeOptions.CompileArchitecture;
import edu.syr.pcpratts.rootbeer.runtime2.cuda.BlockingQueue;
import edu.syr.pcpratts.rootbeer.util.CudaPath;

public class ParallelCompile implements Runnable {

  private BlockingQueue<ParallelCompileJob> m_toCores;
  private BlockingQueue<ParallelCompileJob> m_fromCores;
  
  public ParallelCompile(){
    m_toCores = new BlockingQueue<ParallelCompileJob>();
    m_fromCores = new BlockingQueue<ParallelCompileJob>();
    
    int num_cores = 2;
    for(int i = 0; i < num_cores; ++i){
      Thread thread = new Thread(this);
      thread.setDaemon(true);
      thread.start();
    }
  }
  
  /**
   * @return an array containing compilation results. You can use <tt>is32Bit()</tt> on each element 
   * to determine if it is 32 bit or 64bit code. If compilation for an architecture fails, only the 
   * offending element is returned.
   */
  public CompileResult[] compile(File generated, CudaPath cuda_path, 
    String gencode_options, CompileArchitecture compileArch){
    
    switch (compileArch) {
      case Arch32bit:
        System.out.println("compiling CUDA code for 32bit only...");
        m_toCores.put(new ParallelCompileJob(generated, cuda_path, gencode_options, true));
        break;
      case Arch64bit:
        System.out.println("compiling CUDA code for 64bit only...");
        m_toCores.put(new ParallelCompileJob(generated, cuda_path, gencode_options, false));
        break;
      case Arch32bit64bit:
        System.out.println("compiling CUDA code for 32bit and 64bit...");
        m_toCores.put(new ParallelCompileJob(generated, cuda_path, gencode_options, true));
        m_toCores.put(new ParallelCompileJob(generated, cuda_path, gencode_options, false));
        break;
    }
    
    ParallelCompileJob[] compJobs = new ParallelCompileJob[m_toCores.size()];
    for(int i = 0; i < m_toCores.size(); i++) {
      compJobs[i] = m_fromCores.take();
    }

    List<CompileResult> compResults = new LinkedList<CompileResult>();
    for(ParallelCompileJob j: compJobs) {
      if(j != null) {
        compResults.add(j.getResult());
      }
    }
    
    return compResults.toArray(new CompileResult[compResults.size()]);
  }

  public void run() {
    ParallelCompileJob job = m_toCores.take();
    job.compile();
    m_fromCores.put(job);
  }
  
}
