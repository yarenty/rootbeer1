/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer;

import java.util.Iterator;
import soot.SootClass;
import soot.SootMethod;
import soot.rbclassload.EntryPointDetector;

public class KernelEntryPointDetector implements EntryPointDetector {

  public boolean isEntryPoint(SootMethod sm) {
    if(isKernel(sm)){
      return true;
    }
    if(isRootbeer(sm)){
      return true;
    }
    return false;
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
  
  private boolean isRootbeer(SootMethod sm){
    if(sm.getSignature().equals("<edu.syr.pcpratts.rootbeer.runtime.Rootbeer: void runAll(java.util.List)>")){
      return true;
    }
    return false;
  }
}
