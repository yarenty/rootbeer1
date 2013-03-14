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
import soot.jimple.InvokeExpr;
import soot.jimple.Jimple;
import soot.jimple.ReturnStmt;
import soot.jimple.StringConstant;
import soot.rbclassload.EntryPointDetector;
import soot.rbclassload.MethodSignatureUtil;

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
    
    SootMethod gpu_method = null;
    
    if(isApplication()){
      SootClass prov_class = Scene.v().getSootClass(m_provider);
      SootMethod get_entry_sig = prov_class.getMethod("java.lang.String getEntrySignature()");
      gpu_method = execGetEntrySig(get_entry_sig);
    } else {      
      SootClass prov_class = Scene.v().getSootClass(m_provider);
      SootMethod method = prov_class.getMethodByName("create");
      SootClass kernel_class = searchMethod(method);
      gpu_method = kernel_class.getMethod("void gpuMethod()");
    }
    
    m_signature = gpu_method.getSignature();
    m_initialized = true;
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

  private SootMethod execGetEntrySig(SootMethod method) {
    Body body = method.retrieveActiveBody();
    PatchingChain<Unit> units = body.getUnits();
    Unit last = units.getLast();
    ReturnStmt ret_stmt = (ReturnStmt) last;
    Value op = ret_stmt.getOp();
    StringConstant string_constant = (StringConstant) op;
    String signature = string_constant.value;
    MethodSignatureUtil util = new MethodSignatureUtil();
    util.parse(signature);
    return util.getSootMethod();
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

  public void testEntryPoint(SootMethod sm) {
    if(m_initialized == false){
      init();
    }
    if(sm.getSignature().equals(m_signature)){
      m_entryPoints.add(sm.getSignature());
    }
  }

  public List<String> getEntryPoints() {
    return m_entryPoints;
  }

  private boolean isApplication() {
    if(m_provider.startsWith("edu.syr.pcpratts.rootbeer.testcases.rootbeertest.apps.")){
      return true;
    }
    return false;
  }
}
