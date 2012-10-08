/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.generate.opencl;

import java.util.List;
import soot.Scene;
import soot.SootClass;
import soot.SootMethod;
import soot.Type;

public class IsPolyMorphic {

  public boolean isPoly(SootMethod soot_method, List<Type> hierarchy){ 
    if(hierarchy.size() > 2){
      return true;
    }
    if(hierarchy.size() == 1){
      return false;
    }
    String sub_sig = soot_method.getSubSignature();
    SootClass obj_class = Scene.v().getSootClass("java.lang.Object");
    if(obj_class.declaresMethod(sub_sig)){
      return true;
    }
    return false;
  }
}
