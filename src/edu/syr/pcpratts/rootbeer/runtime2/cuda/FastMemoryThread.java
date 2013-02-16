/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.runtime2.cuda;

public class FastMemoryThread implements Runnable {

  private BlockingQueue<FastMemoryThreadJob> m_toCores;
  private BlockingQueue<FastMemoryThreadJob> m_fromCores;
  
  public FastMemoryThread(){
    m_toCores = new BlockingQueue<FastMemoryThreadJob>();
    m_fromCores = new BlockingQueue<FastMemoryThreadJob>();
    Thread thread = new Thread(this);
    thread.setDaemon(true);
    thread.start();
  }
  
  public void writeArray(FastMemory this_ref, int[] array, long addr, int start, int stop){
    FastMemoryThreadJob job = new FastMemoryThreadJob(this_ref, array, addr, start, stop);
    m_toCores.put(job);
  }
  
  public void join(){
    m_fromCores.take();
  }
  
  public void run() {
    while(true){
      FastMemoryThreadJob job = m_toCores.take();
      job.process();
      m_fromCores.put(job);
    }
  }

  private static class FastMemoryThreadJob {

    private FastMemory m_thisRef;
    private int[] m_array;
    private long m_addr;
    private int m_start;
    private int m_stop;
    
    private FastMemoryThreadJob(FastMemory this_ref, int[] array, long addr, int start, int stop) {
      m_thisRef = this_ref;
      m_array = array;
      m_addr = addr;
      m_start = start;
      m_stop = stop;
    }

    private void process() {
      m_thisRef.doWriteIntArrayEx(m_array, m_addr, m_start, m_stop);
    }
  }

}
