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

import soot.*;
import soot.jimple.InvokeExpr;
import soot.jimple.Jimple;
import soot.jimple.ReturnStmt;
import soot.jimple.StringConstant;
import soot.rtaclassload.EntryMethodTester;
import soot.rtaclassload.MethodSignatureUtil;
import soot.rtaclassload.MethodTester;
import soot.rtaclassload.Operand;
import soot.rtaclassload.RTAClass;
import soot.rtaclassload.RTAInstruction;
import soot.rtaclassload.RTAMethod;
import soot.rtaclassload.RTAClassLoader;

public class TestCaseEntryPointDetector implements EntryMethodTester {

  private String testCase;
  private List<String> testCasePackages;
  private String provider;
  private boolean initialized;
  private String kernelSignature;
  private String createSignature;
  private RTAClass kernelClass;
  private RTAClass createClass;
  
  public TestCaseEntryPointDetector(String test_case){
    testCase = test_case;
    testCasePackages = new ArrayList<String>();
    testCasePackages.add("org.trifort.rootbeer.testcases.otherpackage.");
    testCasePackages.add("org.trifort.rootbeer.testcases.otherpackage2.");
    testCasePackages.add("org.trifort.rootbeer.testcases.rootbeertest.");
    testCasePackages.add("org.trifort.rootbeer.testcases.rootbeertest.apps.fastmatrixdebug.");
    testCasePackages.add("org.trifort.rootbeer.testcases.rootbeertest.arraysum.");
    testCasePackages.add("org.trifort.rootbeer.testcases.rootbeertest.baseconversion.");
    testCasePackages.add("org.trifort.rootbeer.testcases.rootbeertest.canonical.");
    testCasePackages.add("org.trifort.rootbeer.testcases.rootbeertest.canonical2.");
    testCasePackages.add("org.trifort.rootbeer.testcases.rootbeertest.exception.");
    testCasePackages.add("org.trifort.rootbeer.testcases.rootbeertest.gpurequired.");
    testCasePackages.add("org.trifort.rootbeer.testcases.rootbeertest.kerneltemplate.");
    testCasePackages.add("org.trifort.rootbeer.testcases.rootbeertest.ofcoarse.");
    testCasePackages.add("org.trifort.rootbeer.testcases.rootbeertest.remaptest.");
    testCasePackages.add("org.trifort.rootbeer.testcases.rootbeertest.serialization.");

    initialized = false;
  }
  
  private void init(){
    if(testCase.contains(".") == false){
      String new_test_case = findTestCaseClass(testCase);
      if(new_test_case == null){
        System.out.println("cannot find test case class: "+testCase);
        System.exit(0);
      }
      testCase = new_test_case;
    }
    provider = testCase;
    
    createClass = RTAClassLoader.v().getRTAClass(provider);
    RTAMethod createMethod = createClass.findMethodBySubSignature("void create()");
    createSignature = createMethod.getSignature().toString();
    kernelClass = searchMethod(createMethod);
    RTAMethod gpuMethod = kernelClass.findMethodBySubSignature("void gpuMethod()");
    kernelSignature = gpuMethod.getSignature().toString();
    initialized = true;
  }

  public String getProvider() {
    return provider;
  }
  
  private RTAClass searchMethod(RTAMethod method) {
    List<RTAInstruction> instructions = method.getInstructions();
    for(RTAInstruction inst : instructions){
      String name = inst.getName();
      if(name.equals("new")){
        List<Operand> operands = inst.getOperands();
        for(Operand operand : operands){
          if(operand.getType().equals("class_ref") == false){
            continue;
          }
          String class_name = operand.getValue();
          RTAClass rtaClass = RTAClassLoader.v().getRTAClass(class_name);
          List<String> ifaces = rtaClass.getInterfaceStrings();
          for(String iface : ifaces){
            if(iface.equals("org.trifort.rootbeer.runtime.Kernel")){
              return rtaClass;
            }
          }
        }
      }
    }
    return null;
  }
  
  private String findTestCaseClass(String test_case) {
    for(String pkg : testCasePackages){
      String name = pkg + test_case;
      RTAClass existTest = RTAClassLoader.v().getRTAClass(name);
      if(existTest != null){
        return name;
      }
    }
    return null;
  }

  public boolean matches(RTAMethod sm) {
    if(initialized == false){
      init();
    }
    if(sm.getSignature().toString().equals(kernelSignature)){
      return true;
    }
    if(sm.getSignature().toString().equals(createSignature)){
      return true;
    }
    return false;
  }

  @Override
  public Set<String> getNewInvokes() {
    Set<String> ret = new TreeSet<String>();
    ret.add(kernelClass.getName());
    ret.add(createClass.getName());
    ret.addAll(CompilerSetup.getNewInvokes());
    return ret;
  }
}
