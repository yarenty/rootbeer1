/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.runtime;

import java.util.Iterator;
import java.util.List;

public class ResultIterator implements Iterator<Kernel> {

  private Iterator<Kernel> m_currIter;
  private Iterator<Kernel> m_jobsToEnqueue;
  private ParallelRuntime m_runtime;
  private Rootbeer m_rootbeer;

  public ResultIterator(PartiallyCompletedParallelJob partial, ParallelRuntime runtime, Rootbeer rootbeer){
    readPartial(partial);
    m_runtime = runtime;
    m_rootbeer = rootbeer;
  }

  private void readPartial(PartiallyCompletedParallelJob partial){
    List<Kernel> active_jobs = partial.getActiveJobs();
    m_currIter = active_jobs.iterator();
    m_jobsToEnqueue = partial.getJobsToEnqueue();
  }

  public boolean hasNext() {
    if(m_currIter.hasNext())
      return true;
    if(m_jobsToEnqueue.hasNext() == false)
      return false;
    try {
      readPartial(m_runtime.run(m_jobsToEnqueue, m_rootbeer, null));
    } catch(Exception ex){
      ex.printStackTrace();
      return false;
    }
    return m_currIter.hasNext();
  }

  public Kernel next() {
    return m_currIter.next();
  }

  public void remove() {
    throw new UnsupportedOperationException("Not supported yet.");
  }

}
