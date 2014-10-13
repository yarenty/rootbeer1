/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.generate.opencl.fields;

import java.util.Comparator;

import org.trifort.rootbeer.generate.opencl.OpenCLScene;

import soot.SootClass;
import soot.rtaclassload.RTAClassLoader;

public class NumberedTypeSortComparator implements Comparator<SootClass>{

  private boolean m_lowest;
  
  public NumberedTypeSortComparator(boolean lowest_type_num_first) {
    m_lowest = lowest_type_num_first;
  }

  public int compare(SootClass lhs, SootClass rhs) {
    Integer lhs_number = Integer.valueOf(OpenCLScene.v().getTypeNumber(lhs.getName()));
    Integer rhs_number = Integer.valueOf(OpenCLScene.v().getTypeNumber(rhs.getName()));
    
    if(m_lowest){
      return lhs_number.compareTo(rhs_number);
    } else {
      return rhs_number.compareTo(lhs_number);
    }  
  }
  
}