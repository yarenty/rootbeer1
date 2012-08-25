/*
 * Copyright 2012 Ecole Polytechnique de Montreal & Tata Consultancy Services All rights reserved.
 * 
 * The contents of this file are available subject to the terms of the
 * Lesser GNU General Public License (LGPL) Version 2.1.
 * http://www.gnu.org/licenses/lgpl-2.1.html.
 */
package edu.syr.pcpratts.rootbeer.classloader;

import java.util.ArrayList;
import java.util.Collection;
import soot.SootClass;
import soot.SootMethod;

/**
 * Different utilities to detect things in the abstract representation.
 *
 * @author Marc-Andre Laverdiere-Papineau
 */
public class DetectionUtils {

  /**
   * Finds all the constructors of a given class
   * @param sc the soot class object
   * @return a non-null (but potentially empty) collection of constructors
   */
  public static Collection<SootMethod> findConstructors(SootClass sc) {
    ArrayList<SootMethod> returnVal = new ArrayList<SootMethod>();

    for (SootMethod sm : sc.getMethods()) {
      if ("<init>".equals(sm.getName())) {
        returnVal.add(sm);
      }
    }
    return returnVal;
  }

  /**
   * Find all static initializers of a class. 
   * @param sc the soot class object
   * @return a non-null collection object with the static initializer inside, if any was found.
   */
  public static Collection<SootMethod> findStaticInitializers(SootClass sc) {
    ArrayList<SootMethod> returnVal = new ArrayList<SootMethod>();

    for (SootMethod sm : sc.getMethods()) {
      if ("<clinit>".equals(sm.getName())) {
        returnVal.add(sm);
      }
    }
    return returnVal;
  }
}
