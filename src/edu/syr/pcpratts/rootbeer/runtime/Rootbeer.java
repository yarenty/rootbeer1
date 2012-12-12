/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.runtime;

import edu.syr.pcpratts.rootbeer.Configuration;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class Rootbeer implements IRootbeer {

  private IRootbeer m_Rootbeer;
  private List<StatsRow> m_stats;
  
  public Rootbeer(){
    RootbeerFactory factory = new RootbeerFactory();
    m_Rootbeer = factory.create(this);
  }
  
  public void runAll(List<Kernel> jobs) {
    if(jobs.isEmpty()){
      return;
    }
    if(jobs.get(0) instanceof CompiledKernel == false){
      for(Kernel job : jobs){
        job.gpuMethod();
      }
    } else {
      m_stats = new ArrayList<StatsRow>();
      m_Rootbeer.runAll(jobs);
    }
  }

  public Iterator<Kernel> run(Iterator<Kernel> jobs) {
    return m_Rootbeer.run(jobs);
  }
  
  public void addStatsRow(StatsRow row) {
    m_stats.add(row);
  }
  
  public List<StatsRow> getStats(){
    return m_stats;
  }
}
