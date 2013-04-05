/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.entry;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import soot.*;
import soot.rbclassload.HierarchySootClass;
import soot.rbclassload.HierarchySootMethod;
import soot.jimple.InvokeExpr;
import soot.jimple.Jimple;
import soot.jimple.ReturnStmt;
import soot.jimple.StringConstant;
import soot.rbclassload.ClassHierarchy;
import soot.rbclassload.EntryPointDetector;
import soot.rbclassload.MethodSignatureUtil;
import soot.rbclassload.RootbeerClassLoader;

public class TestCaseEntryPointDetector implements EntryPointDetector {

  private String m_testCase;
  private List<SootClass> m_kernels;
  private List<String> m_testCasePackages;
  private String m_provider;
  private boolean m_initialized;
  private String m_signature;
  private List<String> m_entryPoints;
  
  public TestCaseEntryPointDetector(String test_case){
    m_testCase = test_case;
    m_testCasePackages = new ArrayList<String>();
    m_testCasePackages.add("edu.syr.pcpratts.rootbeer.testcases.otherpackage.");
    m_testCasePackages.add("edu.syr.pcpratts.rootbeer.testcases.otherpackage2.");
    m_testCasePackages.add("edu.syr.pcpratts.rootbeer.testcases.rootbeertest.");
    m_testCasePackages.add("edu.syr.pcpratts.rootbeer.testcases.rootbeertest.apps.fastmatrixdebug.");
    m_testCasePackages.add("edu.syr.pcpratts.rootbeer.testcases.rootbeertest.arraysum.");
    m_testCasePackages.add("edu.syr.pcpratts.rootbeer.testcases.rootbeertest.baseconversion.");
    m_testCasePackages.add("edu.syr.pcpratts.rootbeer.testcases.rootbeertest.exception.");
    m_testCasePackages.add("edu.syr.pcpratts.rootbeer.testcases.rootbeertest.gpurequired.");
    m_testCasePackages.add("edu.syr.pcpratts.rootbeer.testcases.rootbeertest.kerneltemplate.");
    m_testCasePackages.add("edu.syr.pcpratts.rootbeer.testcases.rootbeertest.ofcoarse.");
    m_testCasePackages.add("edu.syr.pcpratts.rootbeer.testcases.rootbeertest.remaptest.");
    m_testCasePackages.add("edu.syr.pcpratts.rootbeer.testcases.rootbeertest.serialization.");

    m_initialized = false;
    m_entryPoints = new ArrayList<String>();
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
    
    ClassHierarchy class_hierarchy = RootbeerClassLoader.v().getClassHierarchy();
    HierarchySootClass prov_class = class_hierarchy.getHierarchySootClass(m_provider);
    HierarchySootMethod create_method = prov_class.findMethodByName("create");
    HierarchySootClass kernel_class = searchMethod(create_method);
    HierarchySootMethod gpu_method = kernel_class.findMethodBySubSignature("void gpuMethod()");
    m_signature = gpu_method.getSignature();
    m_initialized = true;
  }

  public String getProvider() {
    return m_provider;
  }
    
  private HierarchySootClass searchMethod(HierarchySootMethod method) {
    return RootbeerClassLoader.v().getClassHierarchy().getHierarchySootClass("edu.syr.pcpratts.rootbeer.testcases.rootbeertest.serialization.CovarientRunOnGpu");
    
    /*
    HierarchySootClass hclass = method.getHierarchySootClass();
    
    Instruction inst = method.getInstructions();
    */
    
    /*
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
    */
    //return null;
  }

  private String findTestCaseClass(String test_case) {
    for(String pkg : m_testCasePackages){
      String name = pkg + test_case;
      if(RootbeerClassLoader.v().getClassHierarchy().containsClass(name)){
        return name;
      }
    }
    return null;
  }

  public void testEntryPoint(HierarchySootMethod sm) {
    if(m_initialized == false){
      init();
    }
    if(sm.getSignature().equals(m_signature)){
      if(m_entryPoints.contains(sm.getSignature()) == false){
        m_entryPoints.add(sm.getSignature());
      }
    }
  }

  public List<String> getEntryPoints() {
    return m_entryPoints;
  }
}
