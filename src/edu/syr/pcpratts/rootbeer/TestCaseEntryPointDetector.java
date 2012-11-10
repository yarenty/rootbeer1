/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import soot.*;
import soot.jimple.InvokeExpr;
import soot.rbclassload.EntryPointDetector;

public class TestCaseEntryPointDetector implements EntryPointDetector {

  private String m_testCase;
  private List<SootClass> m_kernels;
  private List<String> m_testCasePackages;
  private String m_provider;
  private boolean m_initialized;
  private String m_signature;
  
  public TestCaseEntryPointDetector(String test_case){
    m_testCase = test_case;
    m_testCasePackages = new ArrayList<String>();
    m_testCasePackages.add("edu.syr.pcpratts.rootbeer.testcases.otherpackage.");
    m_testCasePackages.add("edu.syr.pcpratts.rootbeer.testcases.otherpackage2.");
    m_testCasePackages.add("edu.syr.pcpratts.rootbeer.testcases.rootbeertest.");
    m_testCasePackages.add("edu.syr.pcpratts.rootbeer.testcases.rootbeertest.arraysum.");
    m_testCasePackages.add("edu.syr.pcpratts.rootbeer.testcases.rootbeertest.baseconversion.");
    m_testCasePackages.add("edu.syr.pcpratts.rootbeer.testcases.rootbeertest.exception.");
    m_testCasePackages.add("edu.syr.pcpratts.rootbeer.testcases.rootbeertest.gpurequired.");
    m_testCasePackages.add("edu.syr.pcpratts.rootbeer.testcases.rootbeertest.ofcoarse.");
    m_testCasePackages.add("edu.syr.pcpratts.rootbeer.testcases.rootbeertest.remaptest.");
    m_testCasePackages.add("edu.syr.pcpratts.rootbeer.testcases.rootbeertest.serialization.");

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
    
    SootClass prov_class = Scene.v().getSootClass(m_provider);
    SootMethod method = prov_class.getMethodByName("create");
    SootClass kernel_class = searchMethod(method);
    SootMethod gpu_method = kernel_class.getMethodByName("gpuMethod");
    m_signature = gpu_method.getSignature();
    m_initialized = true;
  }
  
  public boolean isEntryPoint(SootMethod sm) {
    if(m_initialized == false){
      init();
    }
    if(sm.getSignature().equals(m_signature)){
      return true;
    }
    return false;
  }

  public String getProvider() {
    return m_provider;
  }

  private SootClass searchMethod(SootMethod method) {
    Body body = method.retrieveActiveBody();
    List<ValueBox> boxes = body.getUseAndDefBoxes();
    for(ValueBox box : boxes){
      Value value = box.getValue();
      if(value instanceof InvokeExpr){
        InvokeExpr expr = (InvokeExpr) value;
        SootClass to_call = expr.getMethodRef().declaringClass();
        Iterator<SootClass> iter = to_call.getInterfaces().iterator();
        while(iter.hasNext()){
          SootClass iface = iter.next();
          if(iface.getName().equals("edu.syr.pcpratts.rootbeer.runtime.Kernel")){
            return to_call;
          }
        }
      }
    }
    return null;
  }

  private String findTestCaseClass(String test_case) {
    for(String pkg : m_testCasePackages){
      String name = pkg + test_case;
      if(Scene.v().containsClass(name)){
        return name;
      }
    }
    return null;
  }
}
