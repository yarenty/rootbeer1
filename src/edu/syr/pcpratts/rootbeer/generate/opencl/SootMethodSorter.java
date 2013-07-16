/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.generate.opencl;

import java.util.Comparator;
import soot.SootMethod;

public class SootMethodSorter implements Comparator<SootMethod> {

  public int compare(SootMethod lhs, SootMethod rhs) {
    String sig1 = lhs.getSignature();
    String sig2 = rhs.getSignature();
    return sig1.compareTo(sig2);
  }

}
