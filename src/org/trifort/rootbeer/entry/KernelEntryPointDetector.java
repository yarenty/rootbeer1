/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.entry;

import java.util.Set;
import java.util.TreeSet;

import soot.rtaclassload.EntryMethodTester;
import soot.rtaclassload.MethodSignature;
import soot.rtaclassload.RTAClass;
import soot.rtaclassload.RTAClassLoader;
import soot.rtaclassload.RTAMethod;
import soot.rtaclassload.RTAType;
import soot.rtaclassload.StringNumbers;

public class KernelEntryPointDetector implements EntryMethodTester {

  private Set<String> newInvokes;
  
  public KernelEntryPointDetector(){
    newInvokes = new TreeSet<String>();
  }
  
  public boolean matches(RTAMethod rtaMethod) {
    String methodName = StringNumbers.v().getString(rtaMethod.getSignature().getMethodName());
    if(methodName.equals("create") == false){
      return false;
    }
    RTAClass rtaClass = rtaMethod.getRTAClass();
    RTAType[] ifaces = rtaClass.getInterfaces();
    for(RTAType iface : ifaces){
      if(iface.toString().equals("org.trifort.rootbeer.test.TestKernelTemplate") ||
         iface.toString().equals("org.trifort.rootbeer.test.TestSerialization") || 
         iface.toString().equals("org.trifort.rootbeer.test.TestException")){
        
        RTAClass kernelClass = KernelFinder.find(rtaMethod);
        RTAMethod[] methods = kernelClass.getMethods();
        for(RTAMethod method : methods){
          MethodSignature sig = method.getSignature();
          if(StringNumbers.v().getString(sig.getMethodName()).equals("<init>")){
            RTAClassLoader.v().addCallGraphLink(sig.toString(), "<org.trifort.rootbeer.runtime.Kernel: void gpuMethod()>");
          }
        }
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
