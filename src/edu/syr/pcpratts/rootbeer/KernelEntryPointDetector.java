/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import soot.SootClass;
import soot.SootMethod;
import soot.rbclassload.EntryPointDetector;

public class KernelEntryPointDetector implements EntryPointDetector {

  /*
    */
   
  private List<String> m_entryPoints;
  
  public KernelEntryPointDetector(){
    m_entryPoints = new ArrayList<String>();
    m_entryPoints.add("<edu.syr.pcpratts.rootbeer.runtime.Rootbeer: void runAll(java.util.List)>");
  }
    
  public void testEntryPoint(SootMethod sm) {
    if(isKernel(sm)){
      m_entryPoints.add(sm.getSignature());
    }
  }

  private boolean isKernel(SootMethod sm){
    if(sm.getSubSignature().equals("void gpuMethod()") == false){
      return false;
    }
    SootClass soot_class = sm.getDeclaringClass();
    Iterator<SootClass> iter = soot_class.getInterfaces().iterator();
    while(iter.hasNext()){
      SootClass iface = iter.next();
      if(iface.getName().equals("edu.syr.pcpratts.rootbeer.runtime.Kernel")){
        return true;
      }
    }
    return false;
  }
  
  public List<String> getEntryPoints(){
    return m_entryPoints;
  }
}
