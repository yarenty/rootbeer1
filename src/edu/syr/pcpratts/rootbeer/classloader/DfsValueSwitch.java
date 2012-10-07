/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.classloader;

import java.util.*;
import soot.*;
import soot.jimple.*;

public class DfsValueSwitch implements JimpleValueSwitch {

  private Set<Type> m_types;
  private Set<SootMethodRef> m_methods;
  private Set<SootFieldRef> m_fields;
  
  public void run(SootMethod method) {
    m_types = new HashSet<Type>();
    m_methods = new HashSet<SootMethodRef>();
    m_fields = new HashSet<SootFieldRef>();    
    
    addType(method.getReturnType());
    List<Type> param_types = method.getParameterTypes();
    for(Type param_type : param_types){
      addType(param_type);
    }
    
    SootClass soot_class = method.getDeclaringClass();
    FastWholeProgram.v().getResolver().resolveClass(soot_class.getName(), SootClass.BODIES);
    
    Body body = method.retrieveActiveBody();
    List<ValueBox> boxes = body.getUseAndDefBoxes();
    for(ValueBox box : boxes){
      Value value = box.getValue();
      value.apply(this);
    }
  }
  
  public Set<Type> getTypes(){
    return m_types;
  }
  
  public Set<SootMethodRef> getMethodRefs(){
    return m_methods;
  }
  
  public Set<SootFieldRef> getFieldRefs(){
    return m_fields;
  }
  
  public void addType(Type type){
    if(m_types.contains(type) == false){
      m_types.add(type);
    }
  }
  
  public void addMethodRef(SootMethodRef ref){
    if(m_methods.contains(ref) == false){
      m_methods.add(ref);
    }
  }
  
  public void addFieldRef(SootFieldRef ref){
    if(m_fields.contains(ref) == false){
      m_fields.add(ref);
    }
  }
  
  public void caseLocal(Local local) {
    addType(local.getType());
  }

  public void caseDoubleConstant(DoubleConstant dc) {
    addType(dc.getType());
  }

  public void caseFloatConstant(FloatConstant fc) {
    addType(fc.getType());
  }

  public void caseIntConstant(IntConstant ic) {
    addType(ic.getType());
  }

  public void caseLongConstant(LongConstant lc) {
    addType(lc.getType());
  }

  public void caseNullConstant(NullConstant nc) {
    addType(nc.getType());
  }

  public void caseStringConstant(StringConstant sc) {
    addType(sc.getType());
  }

  public void caseClassConstant(ClassConstant cc) {
    addType(cc.getType());
  }

  public void defaultCase(Object o) {
  }

  public void caseAddExpr(AddExpr ae) {
    addType(ae.getType());
  }

  public void caseAndExpr(AndExpr ae) {
    addType(ae.getType());
  }

  public void caseCmpExpr(CmpExpr ce) {
    addType(ce.getType());
  }

  public void caseCmpgExpr(CmpgExpr ce) {
    addType(ce.getType());
  }

  public void caseCmplExpr(CmplExpr ce) {
    addType(ce.getType());
  }

  public void caseDivExpr(DivExpr de) {
    addType(de.getType());
  }

  public void caseEqExpr(EqExpr eqexpr) {
    addType(eqexpr.getType());
  }

  public void caseNeExpr(NeExpr neexpr) {
    addType(neexpr.getType());
  }

  public void caseGeExpr(GeExpr geexpr) {
    addType(geexpr.getType());
  }

  public void caseGtExpr(GtExpr gtexpr) {
    addType(gtexpr.getType());
  }

  public void caseLeExpr(LeExpr leexpr) {
    addType(leexpr.getType());
  }

  public void caseLtExpr(LtExpr ltexpr) {
    addType(ltexpr.getType());
  }

  public void caseMulExpr(MulExpr me) {
    addType(me.getType());
  }

  public void caseOrExpr(OrExpr orexpr) {
    addType(orexpr.getType());
  }

  public void caseRemExpr(RemExpr re) {
    addType(re.getType());
  }

  public void caseShlExpr(ShlExpr se) {
    addType(se.getType());
  }

  public void caseShrExpr(ShrExpr se) {
    addType(se.getType());
  }

  public void caseUshrExpr(UshrExpr ue) {
    addType(ue.getType());
  }

  public void caseSubExpr(SubExpr se) {
    addType(se.getType());
  }

  public void caseXorExpr(XorExpr xe) {
    addType(xe.getType());
  }

  public void caseInterfaceInvokeExpr(InterfaceInvokeExpr iie) {
    addMethodRef(iie.getMethodRef());
  }

  public void caseSpecialInvokeExpr(SpecialInvokeExpr sie) {
    addMethodRef(sie.getMethodRef());
  }

  public void caseStaticInvokeExpr(StaticInvokeExpr sie) {
    addMethodRef(sie.getMethodRef());
  }

  public void caseVirtualInvokeExpr(VirtualInvokeExpr vie) {
    addMethodRef(vie.getMethodRef());
  }

  public void caseDynamicInvokeExpr(DynamicInvokeExpr die) {
  }

  public void caseCastExpr(CastExpr ce) {
    addType(ce.getCastType());
  }

  public void caseInstanceOfExpr(InstanceOfExpr ioe) {
    addType(ioe.getCheckType());
  }

  public void caseNewArrayExpr(NewArrayExpr nae) {
    addType(nae.getType());
  }

  public void caseNewMultiArrayExpr(NewMultiArrayExpr nmae) {
    addType(nmae.getType());
  }

  public void caseNewExpr(NewExpr ne) {
    addType(ne.getBaseType());
  }

  public void caseLengthExpr(LengthExpr le) {
    addType(le.getType());
  }

  public void caseNegExpr(NegExpr ne) {
    addType(ne.getType());
  }

  public void caseArrayRef(ArrayRef ar) {
    addType(ar.getType());
  }

  public void caseStaticFieldRef(StaticFieldRef sfr) {
    addType(sfr.getField().getType());
    addFieldRef(sfr.getFieldRef());
  }

  public void caseInstanceFieldRef(InstanceFieldRef ifr) {
    addType(ifr.getBase().getType());
    addFieldRef(ifr.getFieldRef());
  }

  public void caseParameterRef(ParameterRef pr) {
    addType(pr.getType());
  }

  public void caseCaughtExceptionRef(CaughtExceptionRef cer) {
    addType(cer.getType());
  }

  public void caseThisRef(ThisRef tr) {
    addType(tr.getType());
  }
}
