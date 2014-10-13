/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.entry;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import soot.SootClass;
import soot.SootMethod;
import soot.rtaclassload.EntryMethodTester;
import soot.rtaclassload.MethodTester;
import soot.rtaclassload.RTAClass;
import soot.rtaclassload.RTAMethod;

public class KernelEntryPointDetector implements EntryMethodTester {

  private boolean runTests;
  private Set<String> newInvokes;
  
  public KernelEntryPointDetector(boolean run_tests){
    runTests = run_tests;
    newInvokes = new TreeSet<String>();
  }
  
  public boolean matches(RTAMethod sm) {
    if(sm.getSignature().getSubSignatureString().equals("void gpuMethod()") == false){
      return false;
    }
    RTAClass rtaClass = sm.getRTAClass();
    if(runTests == false){
      if(rtaClass.getName().startsWith("org.trifort.rootbeer.testcases.")){
        return false;
      }
    }
    Iterator<String> iter = rtaClass.getInterfaceStrings().iterator();
    while(iter.hasNext()){
      String iface = iter.next();
      if(iface.equals("org.trifort.rootbeer.runtime.Kernel")){
        newInvokes.add(rtaClass.getName());
        return true;
      }
    }
    return false;
  }

  @Override
  public Set<String> getNewInvokes() {
    return newInvokes;
  }
}
