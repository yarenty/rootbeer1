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
import soot.rbclassload.EntryMethodTester;
import soot.rbclassload.MethodSignatureUtil;
import soot.rbclassload.MethodTester;
import soot.rbclassload.Operand;
import soot.rbclassload.RTAClass;
import soot.rbclassload.RTAInstruction;
import soot.rbclassload.RTAMethod;
import soot.rbclassload.RootbeerClassLoader;

public class TestCaseEntryPointDetector implements EntryMethodTester {

  private String m_testCase;
  private List<SootClass> m_kernels;
  private List<String> m_testCasePackages;
  private String m_provider;
  private boolean m_initialized;
  private String m_signature;
  private RTAClass kernelClass;
  
  public TestCaseEntryPointDetector(String test_case){
    m_testCase = test_case;
    m_testCasePackages = new ArrayList<String>();
    m_testCasePackages.add("org.trifort.rootbeer.testcases.otherpackage.");
    m_testCasePackages.add("org.trifort.rootbeer.testcases.otherpackage2.");
    m_testCasePackages.add("org.trifort.rootbeer.testcases.rootbeertest.");
    m_testCasePackages.add("org.trifort.rootbeer.testcases.rootbeertest.apps.fastmatrixdebug.");
    m_testCasePackages.add("org.trifort.rootbeer.testcases.rootbeertest.arraysum.");
    m_testCasePackages.add("org.trifort.rootbeer.testcases.rootbeertest.baseconversion.");
    m_testCasePackages.add("org.trifort.rootbeer.testcases.rootbeertest.canonical.");
    m_testCasePackages.add("org.trifort.rootbeer.testcases.rootbeertest.canonical2.");
    m_testCasePackages.add("org.trifort.rootbeer.testcases.rootbeertest.exception.");
    m_testCasePackages.add("org.trifort.rootbeer.testcases.rootbeertest.gpurequired.");
    m_testCasePackages.add("org.trifort.rootbeer.testcases.rootbeertest.kerneltemplate.");
    m_testCasePackages.add("org.trifort.rootbeer.testcases.rootbeertest.ofcoarse.");
    m_testCasePackages.add("org.trifort.rootbeer.testcases.rootbeertest.remaptest.");
    m_testCasePackages.add("org.trifort.rootbeer.testcases.rootbeertest.serialization.");

    m_initialized = false;
  }
  
  private void init(){
    if(m_testCase.contains(".") == false){
      String new_test_case = findTestCaseClass(m_testCase);
      if(new_test_case == null){
        System.out.println("cannot find test case class: "+m_testCase);
        System.exit(0);
      }
      m_testCase = new_test_case;
    }
    m_provider = m_testCase;
    
    RTAClass provClass = RootbeerClassLoader.v().getRTAClass(m_provider);
    RTAMethod createMethod = provClass.findMethodByName("create");
    System.out.println(createMethod.getSignature().toString());
    kernelClass = searchMethod(createMethod);
    RTAMethod gpuMethod = kernelClass.findMethodBySubSignature("void gpuMethod()");
    m_signature = gpuMethod.getSignature().toString();
    m_initialized = true;
  }

  public String getProvider() {
    return m_provider;
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
          RTAClass rtaClass = RootbeerClassLoader.v().getRTAClass(class_name);
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
    for(String pkg : m_testCasePackages){
      String name = pkg + test_case;
      RTAClass existTest = RootbeerClassLoader.v().getRTAClass(name);
      if(existTest != null){
        return name;
      }
    }
    return null;
  }

  public boolean matches(RTAMethod sm) {
    if(m_initialized == false){
      init();
    }
    if(sm.getSignature().toString().equals(m_signature)){
      return true;
    }
    return false;
  }

  @Override
  public Set<String> getNewInvokes() {
    Set<String> ret = new TreeSet<String>();
    ret.add(kernelClass.getName());
    return ret;
  }
}
