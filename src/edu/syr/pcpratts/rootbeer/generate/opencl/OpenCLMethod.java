/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.generate.opencl;

import edu.syr.pcpratts.rootbeer.entry.DontDfsMethods;
import edu.syr.pcpratts.rootbeer.generate.bytecode.StaticOffsets;
import edu.syr.pcpratts.rootbeer.generate.opencl.body.MethodJimpleValueSwitch;
import edu.syr.pcpratts.rootbeer.generate.opencl.body.OpenCLBody;
import edu.syr.pcpratts.rootbeer.generate.opencl.tweaks.Tweaks;
import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import soot.*;
import soot.jimple.InstanceInvokeExpr;
import soot.jimple.SpecialInvokeExpr;
import soot.jimple.StaticInvokeExpr;
import soot.options.Options;
import soot.rbclassload.ClassHierarchy;
import soot.rbclassload.MethodSignatureUtil;
import soot.rbclassload.RootbeerClassLoader;

/**
 * Represents an OpenCL function. 
 * @author pcpratts
 */
public class OpenCLMethod {
  private final SootMethod m_sootMethod;
  private SootClass m_sootClass;
  private Set<String> m_dontMangleMethods;
  private Set<String> m_emitUnmangled;
  private MethodSignatureUtil m_util;
  
  public OpenCLMethod(SootMethod soot_method, SootClass soot_class){
    m_sootMethod = soot_method;
    m_sootClass = soot_class;
    m_util = new MethodSignatureUtil();
    createDontMangleMethods();
  }
  
  public String getReturnString(){
    StringBuilder ret = new StringBuilder();
    if(isConstructor()){
      ret.append("int");
    } else {
      OpenCLType return_type = new OpenCLType(m_sootMethod.getReturnType());
      ret.append(return_type.getCudaTypeString());
    }
    return ret.toString();
  }
  
  private String getRestOfArgumentListStringInternal(){
    StringBuilder ret = new StringBuilder();
    List args = m_sootMethod.getParameterTypes();
    
    if(args.size() != 0)
      ret.append(", ");
    
    for(int i = 0; i < args.size(); ++i){
      Type curr_arg = (Type) args.get(i);
      OpenCLType parameter_type = new OpenCLType(curr_arg);
      ret.append(parameter_type.getCudaTypeString());
      ret.append(" parameter" + Integer.toString(i));
      if(i < args.size()-1)
        ret.append(", ");
    }
    ret.append(", int * exception");
    ret.append(")");
    return ret.toString();
  }
  
  private String getArgumentListStringInternal(boolean override_ctor){
    StringBuilder ret = new StringBuilder();
    ret.append("(");

    String address_space_qual = Tweaks.v().getGlobalAddressSpaceQualifier();
    if(isConstructor() == true){
      ret.append(address_space_qual+" char * gc_info");
    } else if((isConstructor() == false || override_ctor == true) && m_sootMethod.isStatic() == false){
      ret.append(address_space_qual+" char * gc_info, int thisref");
    } else {
      ret.append(address_space_qual+" char * gc_info");
    }
    
    ret.append(getRestOfArgumentListStringInternal());
    return ret.toString();
  }

  public String getArgumentListString(boolean ctor_body){
    if(ctor_body){
      String address_space_qual = Tweaks.v().getGlobalAddressSpaceQualifier();
      String ret = "("+address_space_qual+" char * gc_info, int thisref";
      ret += getRestOfArgumentListStringInternal();
      return ret;
    } else {
      return getArgumentListStringInternal(false);
    }
  }

  public String getArgumentListStringPolymorphic(){
    return getArgumentListStringInternal(true);
  }

  private String getMethodDecl(boolean ctor_body){
    StringBuilder ret = new StringBuilder();
    ret.append(Tweaks.v().getDeviceFunctionQualifier()+" ");
    if(ctor_body){
      ret.append("void");
    } else {
      ret.append(getReturnString());
    }
    ret.append(" ");
    ret.append(getPolymorphicNameInternal(ctor_body));
    ret.append(getArgumentListString(ctor_body));
    return ret.toString();
  }
  
  public String getMethodPrototype(){
    String ret = getMethodDecl(false)+";\n";
    if(isConstructor()){
      ret += getMethodDecl(true)+";\n";
    }
    return ret;
  }

  private boolean isLinux(){
    String s = File.separator;
    if(s.equals("/")){
      return true;
    }
    return false;
  }
  
  private String synchronizedEnter(){
    String prefix = Options.v().rbcl_remap_prefix();    
    if(Options.v().rbcl_remap_all() == false){
      prefix = "";
    }
    
    String ret = "";
    ret += "int id;\n";
    ret += "char * mem;\n";
    ret += "char * trash;\n";
    ret += "char * mystery;\n";
    ret += "int count;\n";
    ret += "int old;\n";
    ret += "char * thisref_synch_deref;\n";
    if(m_sootMethod.isStatic() == false){
      ret += "if(thisref == -1){\n";
      SootClass null_ptr = Scene.v().getSootClass(prefix+"java.lang.NullPointerException");
      ret += "  *exception = "+RootbeerClassLoader.v().getClassNumber(null_ptr)+";\n";
      if(returnsAValue()){
        ret += "  return 0;\n";
      } else {
        ret += "  return;\n";
      }
      ret += "}\n";
    }
    ret += "id = getThreadId();\n";
    StaticOffsets static_offsets = new StaticOffsets();
    int junk_index = static_offsets.getEndIndex() - 4;
    int mystery_index = junk_index - 4;
    if(m_sootMethod.isStatic()){
      int offset = static_offsets.getIndex(m_sootClass);
      ret += "mem = edu_syr_pcpratts_gc_deref(gc_info, 0);\n";
      ret += "trash = mem + "+junk_index+";\n";
      ret += "mystery = mem + "+mystery_index+";\n";
      ret += "mem += "+offset+";\n";
    } else {
      ret += "mem = edu_syr_pcpratts_gc_deref(gc_info, thisref);\n";
      ret += "trash = edu_syr_pcpratts_gc_deref(gc_info, 0) + "+junk_index+";\n";
      ret += "mystery = trash - 8;\n";
      ret += "mem += 12;\n";
    }
    ret += "count = 0;\n";
    ret += "while(count < 100){\n";
    ret += "  old = atomicCAS((int *) mem, -1 , id);\n";
    ret += "  *((int *) trash) = old;\n";
    if(isLinux()){
      ret += "  if(old == -1 || old == id){\n";
    } else {
      ret += "  if(old != -1 && old != id){\n";
      ret += "    count++;\n";
      ret += "    if(count > 50 || (*((int *) mystery)) == 0){\n";
      ret += "      count = 0;\n";
      ret += "    }\n";
      ret += "  } else {\n"; 
    }    
    
    //adding this in makes the WhileTrueTest pass.
    //for some reason the first write to memory doesn't work well inside a sync block.
    if(m_sootMethod.isStatic() == false){
      ret += "  if ( thisref ==-1 ) { \n";
      ret += "    * exception = 11;\n";
      ret += "  }\n";

      ret += "  if ( * exception != 0 ) {\n";
      ret += "    edu_syr_pcpratts_exitMonitorMem ( gc_info , mem , old ) ;\n";
      if(returnsAValue()){
        ret += "    return 0;\n";
      } else {
        ret += "    return;\n";
      }
      ret += "  }\n";

      ret += "  thisref_synch_deref = edu_syr_pcpratts_gc_deref ( gc_info , thisref );\n";
      ret += "  * ( ( int * ) & thisref_synch_deref [ 16 ] ) = 20 ;\n";
    }
    return ret;
  }
  
  public String getMethodBody(){
    StringBuilder ret = new StringBuilder();
    if(shouldEmitBody()){
      ret.append(getMethodDecl(false)+"{\n");
      try {
        if(methodIsRuntimeBasicBlockRun() == false){
          OpenCLBody ocl_body = new OpenCLBody(m_sootMethod, isConstructor());
          ret.append(ocl_body.getLocals());
          if(isSynchronized()){
            ret.append(synchronizedEnter()); 
          }
          try {
          ret.append(ocl_body.getBodyNoLocals());
          } catch(Exception ex){
            ex.printStackTrace();
            System.out.println("soot_method: "+m_sootMethod.getSignature());
            System.exit(1);
          }
          if(isSynchronized()){
            if(isLinux()){
              ret.append("  } else {");
              ret.append("    count++;\n");
              ret.append("    if(count > 50 || (*((int *) mystery)) == 0){\n");
              ret.append("      count = 0;\n");
              ret.append("    }\n");
              ret.append("  }\n"); 
              ret.append("}\n"); 
            } else {
              ret.append("  }\n");
              ret.append("}\n");
            }
          }
          if(returnsAValue()){
            ret.append("return 0;");
          }
        }
      } catch(RuntimeException ex){
        System.out.println("error creating method body: "+m_sootMethod.getSignature());
        OpenCLMethod ocl_method = new OpenCLMethod(m_sootMethod, m_sootClass);
        if(ocl_method.returnsAValue())
          ret.append("return 0;\n");
        else
          ret.append("\n");
      }
      ret.append("}\n");
      if(isConstructor()){
        ret.append(getMethodDecl(true)+"{\n"); 
        OpenCLBody ocl_body = new OpenCLBody(m_sootMethod.retrieveActiveBody());
        ret.append(ocl_body.getBody());
        ret.append("}\n");
      }
    }
    return ret.toString();
  }
  
  public String getConstructorBodyInvokeString(SpecialInvokeExpr arg0){
    StringBuilder ret = new StringBuilder();

    ret.append(getPolymorphicNameInternal(true) +"(");
    List args = arg0.getArgs();
    List<String> args_list = new ArrayList<String>();
    args_list.add("gc_info");
    args_list.add("thisref");
    
    for(int i = 0; i < args_list.size() - 1; ++i){
      ret.append(args_list.get(i));
      ret.append(",\n ");
    }
    if(args_list.size() > 0){
      ret.append(args_list.get(args_list.size()-1));
      if(args.size() > 0)
        ret.append(",\n ");
    }
    
    MethodJimpleValueSwitch quick_value_switch = new MethodJimpleValueSwitch(ret);
    for(int i = 0; i < args.size(); ++i){
      Value arg = (Value) args.get(i);
      arg.apply(quick_value_switch);
      if(i < args.size() - 1)
        ret.append(",\n ");
    }
    ret.append(", exception");
    ret.append(")");
    
    return ret.toString();
  }

  public String getInstanceInvokeString(InstanceInvokeExpr arg0){
    IsPolymorphic is_polymorphic = new IsPolymorphic();
    if(is_polymorphic.test(arg0.getMethod(), arg0 instanceof SpecialInvokeExpr)){
      return writeInstanceInvoke(arg0, "invoke_", is_polymorphic.getBaseMethod().getDeclaringClass().getType());
    } else {
      return writeInstanceInvoke(arg0, "", m_sootClass.getType());
    }
  }

  public String getStaticInvokeString(StaticInvokeExpr expr){
    StringBuilder ret = new StringBuilder();

    ret.append(getPolymorphicName()+"(");
    List args = expr.getArgs();
    List<String> args_list = new ArrayList<String>();
    args_list.add("gc_info");

    for(int i = 0; i < args_list.size() - 1; ++i){
      ret.append(args_list.get(i));
      ret.append(", ");
    }
    if(args_list.size() > 0){
      ret.append(args_list.get(args_list.size()-1));
      if(args.size() > 0)
        ret.append(", ");
    }
    MethodJimpleValueSwitch quick_value_switch = new MethodJimpleValueSwitch(ret);
    for(int i = 0; i < args.size(); ++i){
      Value arg = (Value) args.get(i);
      arg.apply(quick_value_switch);
      if(i < args.size() - 1)
        ret.append(", ");
    }
    ret.append(", exception");
    ret.append(")");
    return ret.toString();
  }

  private String writeInstanceInvoke(InstanceInvokeExpr arg0, String method_prefix, Type type){
    if(type instanceof RefType == false){
      throw new RuntimeException("please report bug in OpenCLMethod.writeInstanceInvoke");
    }
    RefType ref_type = (RefType) type;
    OpenCLMethod corrected_this = new OpenCLMethod(m_sootMethod, ref_type.getSootClass());
    StringBuilder ret = new StringBuilder();
    Value base = arg0.getBase();
    if(base instanceof Local == false)
      throw new UnsupportedOperationException("How do we handle an invoke on a non loca?");
    Local local = (Local) base;
    if(isConstructor()){
      ret.append("edu_syr_pcpratts_gc_assign (gc_info, \n&"+local.getName()+", ");
    }

    String function_name = method_prefix+corrected_this.getPolymorphicName();
    ret.append(function_name+"(");
    List args = arg0.getArgs();
    List<String> args_list = new ArrayList<String>();
    args_list.add("gc_info");
    
    //write the thisref
    if(isConstructor() == false)
      args_list.add(local.getName());

    for(int i = 0; i < args_list.size() - 1; ++i){
      ret.append(args_list.get(i));
      ret.append(",\n ");
    }
    if(args_list.size() > 0){
      ret.append(args_list.get(args_list.size()-1));
      if(args.size() > 0)
        ret.append(",\n ");
    }
    
    MethodJimpleValueSwitch quick_value_switch = new MethodJimpleValueSwitch(ret);
    for(int i = 0; i < args.size(); ++i){
      Value arg = (Value) args.get(i);
      arg.apply(quick_value_switch);
      if(i < args.size() - 1)
        ret.append(",\n ");
    }
    ret.append(", exception");
    ret.append(")");
    
    if(isConstructor()){
      ret.append(")");
    }

    return ret.toString();
  }

  public boolean isConstructor(){
    String method_name = m_sootMethod.getName();
    if(method_name.equals("<init>"))
      return true;
    return false;
  }
  
  public String getPolymorphicName(){
    return getPolymorphicNameInternal(false);
  }
  
  private String getPolymorphicNameInternal(boolean ctor_body){
    String ret = getBaseMethodName();
    if(ctor_body){
      ret += "_body";  
    }
    String signature = m_sootMethod.getSignature();
    if(m_dontMangleMethods.contains(signature) == false)
      ret += NameMangling.v().mangleArgs(m_sootMethod);
    return ret;
  }

  private String getBaseMethodName(){
    //if we are here and a method is not concrete, it is the case where a 
    //invoke expresion is to an interface pointer and the method is not polymorphic
    if(m_sootMethod.isConcrete() == false){
      //Set<String> virtual_signatures = RootbeerClassLoader.v().getVirtualSignaturesDown(m_sootMethod, null);
      ClassHierarchy class_hierarchy = RootbeerClassLoader.v().getClassHierarchy();
      List<String> virtual_methods = class_hierarchy.getVirtualMethods(m_sootMethod.getSignature());
      
      //double check that we are safe to do the interface remapping
      if(virtual_methods.size() != 1){
        //not safe, go back to normal method
        return getBaseMethodName(m_sootClass, m_sootMethod);
      } else {
        String virt_method = virtual_methods.iterator().next();
        m_util.parse(virt_method);
        SootMethod soot_method = m_util.getSootMethod();
        SootClass soot_class = soot_method.getDeclaringClass();
        return getBaseMethodName(soot_class, soot_method);
      }
    } else {
      return getBaseMethodName(m_sootClass, m_sootMethod);  
    }
  }
  
  private String getBaseMethodName(SootClass soot_class, SootMethod soot_method){
    OpenCLClass ocl_class = new OpenCLClass(soot_class);

    String method_name = soot_method.getName();
    //here I use a certain uuid for init so there is low chance of collisions
    method_name = method_name.replace("<init>", "init"+OpenCLScene.v().getUuid());

    String ret = ocl_class.getName()+"_"+method_name;
    return ret;
  }
  private boolean shouldEmitBody(){
    String signature = m_sootMethod.getSignature();
    if(m_emitUnmangled.contains(signature)){
      return true;
    }
    if(m_dontMangleMethods.contains(signature)){
      return false;
    }
    return true;
  }
  
  @Override
  public String toString(){
    return getPolymorphicName();
  }

  private boolean methodIsRuntimeBasicBlockRun() {
    if(m_sootClass.getName().equals("edu.syr.pcpratts.javaautogpu.runtime.RuntimeBasicBlock") == false)
      return false;
    if(m_sootMethod.getName().equals("run") == false)
      return false;
    return true;
  }

  public boolean returnsAValue() {
    if(isConstructor())
      return true;
    Type t = m_sootMethod.getReturnType();
    if(t instanceof VoidType)
      return false;
    return true;
  }

  public boolean isSynchronized() {
    return m_sootMethod.isSynchronized();
  }
  
  private void createDontMangleMethods() {
    m_dontMangleMethods = new HashSet<String>();
    m_emitUnmangled = new HashSet<String>();
    
    DontDfsMethods dont_dfs_methods = new DontDfsMethods();
    for(String dont_dfs : dont_dfs_methods.get()){
      m_dontMangleMethods.add(dont_dfs);
    }
    m_dontMangleMethods.add("<java.lang.String: void <init>(char[])>");
    m_dontMangleMethods.add("<edu.syr.pcpratts.rootbeer.runtimegpu.GpuException: edu.syr.pcpratts.rootbeer.runtimegpu.GpuException arrayOutOfBounds(int,int,int)>");
  
    m_emitUnmangled.add("<java.lang.String: void <init>(char[])>");
    m_emitUnmangled.add("<edu.syr.pcpratts.rootbeer.runtimegpu.GpuException: edu.syr.pcpratts.rootbeer.runtimegpu.GpuException arrayOutOfBounds(int,int,int)>");
  }

  public String getSignature() {
    return m_sootMethod.getSignature();
  }

  public SootMethod getSootMethod() {
    return m_sootMethod;
  }
}
