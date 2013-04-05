/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.entry;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import soot.SootClass;
import soot.SootMethod;
import soot.rbclassload.HierarchySootClass;
import soot.rbclassload.HierarchySootMethod;
import soot.rbclassload.EntryPointDetector;

public class KernelEntryPointDetector implements EntryPointDetector {
  
  private List<String> m_entryPoints;
  
  public KernelEntryPointDetector(){
    m_entryPoints = new ArrayList<String>();
    //m_entryPoints.add("<edu.syr.pcpratts.rootbeer.runtime.Rootbeer: void runAll(java.util.List)>");
    //m_entryPoints.add("<edu.syr.pcpratts.rootbeer.runtime.Rootbeer: void runAll(edu.syr.pcpratts.rootbeer.runtime.Kernel)>");
  }
    
  public void testEntryPoint(HierarchySootMethod sm) {
    System.out.println("testEntryPoint: ");
    System.out.println("  "+sm.getSignature());
    System.out.println("  "+sm.getSignature());
    if(isKernel(sm)){
      m_entryPoints.add(sm.getSignature());
    }
  }

  private boolean isKernel(HierarchySootMethod sm){
    if(sm.getSubSignature().equals("void gpuMethod()") == false){
      return false;
    }
    HierarchySootClass soot_class = sm.getHierarchySootClass();
    Iterator<String> iter = soot_class.getInterfaces().iterator();
    while(iter.hasNext()){
      String iface = iter.next();
      if(iface.equals("edu.syr.pcpratts.rootbeer.runtime.Kernel")){
        return true;
      }
    }
    return false;
  }
  
  public List<String> getEntryPoints(){
    return m_entryPoints;
  }

}
