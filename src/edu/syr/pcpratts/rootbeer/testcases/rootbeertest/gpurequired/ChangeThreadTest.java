/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.testcases.rootbeertest.gpurequired;

import edu.syr.pcpratts.rootbeer.runtime.Kernel;
import edu.syr.pcpratts.rootbeer.runtime.Rootbeer;
import edu.syr.pcpratts.rootbeer.test.TestSerialization;
import java.util.ArrayList;
import java.util.List;

public class ChangeThreadTest implements TestSerialization {

  private Rootbeer m_rootbeer;
  
  public List<Kernel> create() {
    CreateRootbeerThread creator = new CreateRootbeerThread();
    Thread t = new Thread(creator);
    t.start();
    try { 
      t.join();
    } catch(Exception ex){
      ex.printStackTrace();
    }
    m_rootbeer = creator.getRootbeer();
    
    List<Kernel> ret = new ArrayList<Kernel>();
    for(int i = 0; i < 10; ++i){
      ret.add(new ChangeThreadRunOnGpu());
    }
    return ret;
  }

  public boolean compare(Kernel original, Kernel from_heap) {
    ChangeThreadRunOnGpu lhs = (ChangeThreadRunOnGpu) original;
    ChangeThreadRunOnGpu rhs = (ChangeThreadRunOnGpu) from_heap;
    return lhs.compare(rhs);
  }
  
  private class CreateRootbeerThread implements Runnable {
    private Rootbeer m_rootbeer;
    public void run() {
      m_rootbeer = new Rootbeer();
    }
    public Rootbeer getRootbeer(){
      return m_rootbeer;
    }
  }
}
