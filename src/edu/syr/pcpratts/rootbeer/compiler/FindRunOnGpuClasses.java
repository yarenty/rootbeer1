/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.compiler;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import soot.Scene;
import soot.SootClass;
import soot.util.Chain;

public class FindRunOnGpuClasses {
  
  public List<String> get(List<String> app_classes){
       
    List<String> ret = new ArrayList<String>();
    for(String cls : app_classes){
      try {
        SootClass soot_class = Scene.v().getSootClass(cls);
        if(matches(soot_class)){
          ret.add(soot_class.getName());
        }
      } catch(RuntimeException ex){
        continue;
      }
    }
    return ret;
  }
  
  public boolean matches(SootClass soot_class){
    Chain<SootClass> interfaces = soot_class.getInterfaces();
    Iterator<SootClass> iter = interfaces.iterator();
    while(iter.hasNext()){
      SootClass cls = iter.next();
      if(cls.getName().equals("edu.syr.pcpratts.rootbeer.runtime.Kernel"))
        return true;
    }
    return false;
  }
}
