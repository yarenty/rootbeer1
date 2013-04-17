/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.generate.opencl;

import edu.syr.pcpratts.rootbeer.configuration.RootbeerPaths;
import edu.syr.pcpratts.rootbeer.generate.opencl.fields.OpenCLField;
import edu.syr.pcpratts.rootbeer.generate.bytecode.ReadOnlyTypes;
import edu.syr.pcpratts.rootbeer.generate.codesegment.CodeSegment;
import edu.syr.pcpratts.rootbeer.generate.opencl.fields.CompositeField;
import edu.syr.pcpratts.rootbeer.generate.opencl.fields.CompositeFieldFactory;
import edu.syr.pcpratts.rootbeer.generate.opencl.fields.FieldCodeGeneration;
import edu.syr.pcpratts.rootbeer.generate.opencl.fields.FieldTypeSwitch;
import edu.syr.pcpratts.rootbeer.generate.opencl.fields.OffsetCalculator;
import edu.syr.pcpratts.rootbeer.generate.opencl.tweaks.CompileResult;
import edu.syr.pcpratts.rootbeer.generate.opencl.tweaks.CudaTweaks;
import edu.syr.pcpratts.rootbeer.generate.opencl.tweaks.Tweaks;
import edu.syr.pcpratts.rootbeer.util.ResourceReader;
import soot.rbclassload.MethodSignatureUtil;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import soot.*;
import soot.rbclassload.FieldSignatureUtil;
import soot.rbclassload.NumberedType;
import soot.rbclassload.RootbeerClassLoader;

public class OpenCLScene {
  private static OpenCLScene m_instance;
  private static int m_curentIdent;
  private Map<String, OpenCLClass> m_classes;
  private Set<OpenCLArrayType> m_arrayTypes;
  private CodeSegment m_codeSegment;
  private MethodHierarchies m_methodHierarchies;
  private boolean m_usesGarbageCollector;
  private SootClass m_rootSootClass;
  private int m_endOfStatics;
  private ReadOnlyTypes m_readOnlyTypes;
  private Set<OpenCLInstanceof> m_instanceOfs;
  
  static {
    m_curentIdent = 0;
  }

  public OpenCLScene(){
    m_codeSegment = null;
    m_classes = new LinkedHashMap<String, OpenCLClass>();
    m_arrayTypes = new LinkedHashSet<OpenCLArrayType>();
    m_methodHierarchies = new MethodHierarchies();
    m_instanceOfs = new HashSet<OpenCLInstanceof>();
  }

  public static OpenCLScene v(){
    return m_instance;
  }
  
  public static void setInstance(OpenCLScene scene){
    m_instance = scene;
  }

  public static void releaseV(){
    m_instance = null;
    m_curentIdent++;
  }
  
  public String getIdent(){
    return "" + m_curentIdent;
  }

  public String getUuid(){
    return "ab850b60f96d11de8a390800200c9a66";
  }

  public int getEndOfStatics(){
    return m_endOfStatics;
  }

  public int getClassType(SootClass soot_class){
    return RootbeerClassLoader.v().getClassNumber(soot_class);
  }
  
  public void addMethod(SootMethod soot_method){
    SootClass soot_class = soot_method.getDeclaringClass();

    OpenCLClass ocl_class = getOpenCLClass(soot_class);
    ocl_class.addMethod(new OpenCLMethod(soot_method, soot_class));

    //add the method 
    m_methodHierarchies.addMethod(soot_method);
  }

  public void addArrayType(OpenCLArrayType array_type){
    if(m_arrayTypes.contains(array_type))
      return;
    m_arrayTypes.add(array_type);
  }  
  
  public void addInstanceof(Type type){
    OpenCLInstanceof to_add = new OpenCLInstanceof(type);
    if(m_instanceOfs.contains(to_add) == false){
      m_instanceOfs.add(to_add);
    }
  }

  public OpenCLClass getOpenCLClass(SootClass soot_class){    
    String class_name = soot_class.getName();
    if(m_classes.containsKey(class_name)){
      return m_classes.get(class_name);
    } else {
      OpenCLClass ocl_class = new OpenCLClass(soot_class);
      m_classes.put(class_name, ocl_class);
      return ocl_class;
    }
  }

  public void addField(SootField soot_field){
    SootClass soot_class = soot_field.getDeclaringClass();
    OpenCLClass ocl_class = getOpenCLClass(soot_class);
    ocl_class.addField(new OpenCLField(soot_field, soot_class));
  }

  private String getRuntimeBasicBlockClassName(){
    SootClass soot_class = m_rootSootClass;
    OpenCLClass ocl_class = getOpenCLClass(soot_class);
    return ocl_class.getName();
  }

  private String readCudaCodeFromFile(){
    try {
      BufferedReader reader = new BufferedReader(new FileReader("generated.cu"));
      String ret = "";
      while(true){
        String temp = reader.readLine();
        if(temp == null)
          return ret;
        ret += temp+"\n";
      }
    } catch(Exception ex){
      throw new RuntimeException();
    }
  }

  public void setUsingGarbageCollector(){
    m_usesGarbageCollector = true;
  }

  public boolean getUsingGarbageCollector(){
    return m_usesGarbageCollector;
  }
  
  private void writeTypesToFile(List<NumberedType> types){
    try {
      PrintWriter writer = new PrintWriter(RootbeerPaths.v().getTypeFile());
      for(NumberedType type : types){
        writer.println(type.getNumber()+" "+type.getType().toString());
      }
      writer.flush();
      writer.close();
    } catch(Exception ex){
      ex.printStackTrace();
    }
  }
  
  private String[] makeSourceCode() throws Exception {
    m_usesGarbageCollector = false;
    
    List<NumberedType> types = RootbeerClassLoader.v().getDfsInfo().getNumberedTypes();
    writeTypesToFile(types);
    
    Set<String> methods = RootbeerClassLoader.v().getDfsInfo().getMethods();  
    MethodSignatureUtil util = new MethodSignatureUtil();
    for(String method_sig : methods){
      //System.out.println("OpenCLScene method_sig: "+method_sig);
      util.parse(method_sig);
      SootMethod method = util.getSootMethod();
      addMethod(method);
    }
    List<String> extra_methods = new ArrayList<String>();
    extra_methods.add("<edu.syr.pcpratts.rootbeer.runtimegpu.GpuException: edu.syr.pcpratts.rootbeer.runtimegpu.GpuException arrayOutOfBounds(int,int,int)>");
    extra_methods.add("<edu.syr.pcpratts.rootbeer.runtimegpu.GpuException: void <init>()>");
    for(String extra_method : extra_methods){
      util.parse(extra_method);
      addMethod(util.getSootMethod());
    }
    
    Set<SootField> fields = RootbeerClassLoader.v().getDfsInfo().getFields();
    for(SootField field : fields){
      addField(field);
    }
    FieldSignatureUtil futil = new FieldSignatureUtil();
    List<String> extra_fields = new ArrayList<String>();
    extra_fields.add("<edu.syr.pcpratts.rootbeer.runtimegpu.GpuException: int m_arrayLength>");
    extra_fields.add("<edu.syr.pcpratts.rootbeer.runtimegpu.GpuException: int m_arrayIndex>");
    extra_fields.add("<edu.syr.pcpratts.rootbeer.runtimegpu.GpuException: int m_array>");
    for(String extra_field : extra_fields){
      futil.parse(extra_field);
      addField(futil.getSootField());
    }
    
    Set<ArrayType> array_types = RootbeerClassLoader.v().getDfsInfo().getArrayTypes();
    for(ArrayType array_type : array_types){
      OpenCLArrayType ocl_array_type = new OpenCLArrayType(array_type);
      addArrayType(ocl_array_type);
    }
    
    Set<Type> instanceofs = RootbeerClassLoader.v().getDfsInfo().getInstanceOfs();
    for(Type type : instanceofs){
      addInstanceof(type);
    }
    
    StringBuilder unix_code = new StringBuilder();
    StringBuilder windows_code = new StringBuilder();
    
    String method_protos = methodPrototypesString();
    String gc_string = garbageCollectorString();
    String bodies_string = methodBodiesString();
    
    unix_code.append(headerString(true));
    unix_code.append(method_protos);
    unix_code.append(gc_string);
    unix_code.append(bodies_string);
    unix_code.append(kernelString(true));

    windows_code.append(headerString(false));
    windows_code.append(method_protos);
    windows_code.append(gc_string);
    windows_code.append(bodies_string);
    windows_code.append(kernelString(false));
    
    String cuda_unix = setupEntryPoint(unix_code);
    String cuda_windows = setupEntryPoint(windows_code);
    
    //print out code for debugging
    PrintWriter writer = new PrintWriter(new FileWriter(RootbeerPaths.v().getRootbeerHome()+"generated_unix.cu"));
    writer.println(cuda_unix);
    writer.flush();
    writer.close();
    
    //print out code for debugging
    writer = new PrintWriter(new FileWriter(RootbeerPaths.v().getRootbeerHome()+"generated_windows.cu"));
    writer.println(cuda_windows);
    writer.flush();
    writer.close();
    
    NameMangling.v().writeTypesToFile();
    
    String[] ret = new String[2];
    ret[0] = cuda_unix;
    ret[1] = cuda_windows;
    return ret;
  }

  private String setupEntryPoint(StringBuilder builder){
    String cuda_code = builder.toString();
    String mangle = NameMangling.v().mangle(VoidType.v());
    String replacement = getRuntimeBasicBlockClassName()+"_gpuMethod"+mangle;
    //class names can have $ in them, make them regex safe
    replacement = replacement.replace("$", "\\$");
    cuda_code = cuda_code.replaceAll("%%invoke_run%%", replacement);  
    return cuda_code;
  }
  
  public String[] getOpenCLCode() throws Exception {
    String[] source_code = makeSourceCode();
    return source_code;
  }

  public CompileResult[] getCudaCode() throws Exception {
    String[] source_code = makeSourceCode();
    return new CudaTweaks().compileProgram(source_code[0]);
  }

  private String headerString(boolean unix) throws IOException {
    String specific_path;
    if(unix){
      specific_path = Tweaks.v().getUnixHeaderPath();
    } else {
      specific_path = Tweaks.v().getWindowsHeaderPath();
    }
    if(specific_path == null)
      return "";
    String both_path = Tweaks.v().getBothHeaderPath();
    String both_header = "";
    if(both_path != null){
      both_header = ResourceReader.getResource(both_path);
    }
    String specific_header = ResourceReader.getResource(specific_path);
    return specific_header + "\n" + both_header;
  }
  
  private String kernelString(boolean unix) throws IOException {
    String kernel_path;
    if(unix){
      kernel_path = Tweaks.v().getUnixKernelPath();
    } else {
      kernel_path = Tweaks.v().getWindowsKernelPath();
    }
    String specific_kernel_code = ResourceReader.getResource(kernel_path);
    String both_kernel_code = "";
    String both_kernel_path = Tweaks.v().getBothKernelPath();
    if(both_kernel_path != null){
      both_kernel_code = ResourceReader.getResource(both_kernel_path);
    }
    return both_kernel_code + "\n" + specific_kernel_code;
  }
  
  private String garbageCollectorString() throws IOException {
    String path = Tweaks.v().getGarbageCollectorPath();
    String ret = ResourceReader.getResource(path);
    ret = ret.replace("$$__device__$$", Tweaks.v().getDeviceFunctionQualifier());
    ret = ret.replace("$$__global$$", Tweaks.v().getGlobalAddressSpaceQualifier());
    return ret;
  }

  private String methodPrototypesString(){
    //using a set so duplicates get filtered out.
    Set<String> protos = new HashSet<String>();
    StringBuilder ret = new StringBuilder();
    
    ArrayCopyGenerate arr_generate = new ArrayCopyGenerate();
    protos.add(arr_generate.getProto());
    
    List<OpenCLMethod> methods = m_methodHierarchies.getMethods();
    for(OpenCLMethod method : methods){ 
      protos.add(method.getMethodPrototype());
    }    
    List<OpenCLPolymorphicMethod> poly_methods = m_methodHierarchies.getPolyMorphicMethods();
    for(OpenCLPolymorphicMethod poly_method : poly_methods){
      protos.add(poly_method.getMethodPrototypes());
    }
    FieldCodeGeneration gen = new FieldCodeGeneration();
    protos.add(gen.prototypes(m_classes, m_codeSegment.getReadWriteFieldInspector()));
    for(OpenCLArrayType array_type : m_arrayTypes){
      protos.add(array_type.getPrototypes());
    }
    for(OpenCLInstanceof type : m_instanceOfs){
      protos.add(type.getPrototype());
    }
    Iterator<String> iter = protos.iterator();
    while(iter.hasNext()){
      ret.append(iter.next());
    }
    return ret.toString();
  }

  private String methodBodiesString() throws IOException{
    StringBuilder ret = new StringBuilder();
    if(m_usesGarbageCollector)
      ret.append("#define USING_GARBAGE_COLLECTOR\n");
    //a set is used so duplicates get filtered out
    Set<String> bodies = new HashSet<String>();
    
    ArrayCopyTypeReduction reduction = new ArrayCopyTypeReduction();
    Set<OpenCLArrayType> new_types = reduction.run(m_arrayTypes, m_methodHierarchies);
    
    ArrayCopyGenerate arr_generate = new ArrayCopyGenerate();
    bodies.add(arr_generate.get(new_types));
    
    ObjectCloneGenerate clone_generate = new ObjectCloneGenerate();
    bodies.add(clone_generate.get(m_arrayTypes, m_classes));
    
    List<OpenCLMethod> methods = m_methodHierarchies.getMethods();
    for(OpenCLMethod method : methods){ 
      bodies.add(method.getMethodBody());
    }
    List<OpenCLPolymorphicMethod> poly_methods = m_methodHierarchies.getPolyMorphicMethods();
    for(OpenCLPolymorphicMethod poly_method : poly_methods){
      bodies.add(poly_method.getMethodBodies());
    }
    FieldTypeSwitch type_switch = new FieldTypeSwitch();
    FieldCodeGeneration gen = new FieldCodeGeneration();
    String field_bodies = gen.bodies(m_classes, 
      m_codeSegment.getReadWriteFieldInspector(), type_switch);
    bodies.add(field_bodies);
    for(OpenCLArrayType array_type : m_arrayTypes){
      bodies.add(array_type.getBodies());
    }
    for(OpenCLInstanceof type : m_instanceOfs){
      bodies.add(type.getBody());
    }
    Iterator<String> iter = bodies.iterator();
    ret.append(type_switch.getFunctions());
    while(iter.hasNext()){
      ret.append(iter.next());
    }
    return ret.toString();
  }
  
  public OffsetCalculator getOffsetCalculator(SootClass soot_class){
    CompositeFieldFactory composite_factory = new CompositeFieldFactory();
    List<CompositeField> composites = composite_factory.create(m_classes);
    for(CompositeField composite : composites){
      List<SootClass> classes = composite.getClasses();
      if(classes.contains(soot_class))
        return new OffsetCalculator(composite);
    }
    throw new RuntimeException("Cannot find composite field for soot_class");
  }

  public void addCodeSegment(CodeSegment codeSegment){
    this.m_codeSegment = codeSegment;
    m_rootSootClass = codeSegment.getRootSootClass();    
    m_readOnlyTypes = new ReadOnlyTypes(codeSegment.getRootMethod());
    getOpenCLClass(m_rootSootClass);
  }

  public boolean isArrayLocalWrittenTo(Local local){
    return m_codeSegment.getReadWriteFieldInspector().localRepresentingArrayIsWrittenOnGpu(local);
  }
  
  public ReadOnlyTypes getReadOnlyTypes(){
    return m_readOnlyTypes;
  }

  public boolean isRootClass(SootClass soot_class) {
    return soot_class.getName().equals(m_rootSootClass.getName());
  }

  public Map<String, OpenCLClass> getClassMap(){
    return m_classes;
  }
}
