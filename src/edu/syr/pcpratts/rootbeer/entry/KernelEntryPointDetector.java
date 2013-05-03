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
import soot.rbclassload.MethodTester;

public class KernelEntryPointDetector implements MethodTester {

  public boolean test(HierarchySootMethod sm) {
    if(sm.getSubSignature().equals("void gpuMethod()") == false){
      return false;
    }
    HierarchySootClass soot_class = sm.getHierarchySootClass();
    if(soot_class.getName().startsWith("edu.syr.pcpratts.rootbeer.testcases.")){
      return false;
    }
    Iterator<String> iter = soot_class.getInterfaces().iterator();
    while(iter.hasNext()){
      String iface = iter.next();
      if(iface.equals("edu.syr.pcpratts.rootbeer.runtime.Kernel")){
        return true;
      }
    }
    return false;
  }

}
