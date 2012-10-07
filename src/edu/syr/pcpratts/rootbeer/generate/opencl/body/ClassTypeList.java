/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.generate.opencl.body;

import edu.syr.pcpratts.rootbeer.generate.opencl.OpenCLScene;
import java.util.ArrayList;
import java.util.List;
import soot.SootClass;

public class ClassTypeList {

  public List<Integer> getTypeList(SootClass input_class) {
    List<SootClass> classes = getTrimmedClassHierarchy(input_class);
    List<Integer> ret = new ArrayList<Integer>();
    for(SootClass soot_class : classes){
      int num = OpenCLScene.v().getClassType(soot_class);
      ret.add(num);
    }
    return ret;
  }
  
  public List<SootClass> getTrimmedClassHierarchy(SootClass soot_class){
    List<SootClass> all = OpenCLScene.v().getClassHierarchy(soot_class);
    List<SootClass> ret = new ArrayList<SootClass>();
    for(int i = 0; i < all.size(); ++i){
      SootClass curr = all.get(i);
      ret.add(curr);
      if(curr.equals(soot_class))
        break;
    }
    return ret;
  }
}
