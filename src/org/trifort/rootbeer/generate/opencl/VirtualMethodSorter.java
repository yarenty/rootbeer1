package org.trifort.rootbeer.generate.opencl;

import java.util.Comparator;

import soot.SootClass;
import soot.SootMethod;

public class VirtualMethodSorter implements Comparator<SootMethod> {

  @Override
  public int compare(SootMethod arg0, SootMethod arg1) {
    if(above(arg0, arg1)){
      return -1;
    }
    return 1;
  }

  public boolean above(SootMethod arg0, SootMethod arg1) {
    SootClass leftClass = arg0.getDeclaringClass();
    while(true){
      if(leftClass.hasSuperclass()){
        leftClass = leftClass.getSuperclass();
        if(leftClass == arg1.getDeclaringClass()){
          return true;
        }
      } else {
        return false;
      }
    }
  }
}
