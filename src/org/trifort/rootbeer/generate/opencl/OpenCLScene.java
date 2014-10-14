/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.generate.opencl;

import soot.jimple.NewExpr;
import soot.rtaclassload.MethodSignatureUtil;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.trifort.rootbeer.configuration.Configuration;
import org.trifort.rootbeer.configuration.RootbeerPaths;
import org.trifort.rootbeer.entry.DfsInfo;
import org.trifort.rootbeer.entry.ForcedFields;
import org.trifort.rootbeer.entry.CompilerSetup;
import org.trifort.rootbeer.generate.bytecode.MethodCodeSegment;
import org.trifort.rootbeer.generate.opencl.fields.CompositeField;
import org.trifort.rootbeer.generate.opencl.fields.CompositeFieldFactory;
import org.trifort.rootbeer.generate.opencl.fields.FieldCodeGeneration;
import org.trifort.rootbeer.generate.opencl.fields.FieldTypeSwitch;
import org.trifort.rootbeer.generate.opencl.fields.OffsetCalculator;
import org.trifort.rootbeer.generate.opencl.fields.OpenCLField;
import org.trifort.rootbeer.generate.opencl.tweaks.CompileResult;
import org.trifort.rootbeer.generate.opencl.tweaks.CudaTweaks;
import org.trifort.rootbeer.generate.opencl.tweaks.Tweaks;
import org.trifort.rootbeer.util.ReadFile;
import org.trifort.rootbeer.util.ResourceReader;

import soot.*;
import soot.rtaclassload.FieldSignatureUtil;
import soot.rtaclassload.MethodFieldFinder;
import soot.rtaclassload.MethodSignature;
import soot.rtaclassload.RTAClassLoader;
import soot.rtaclassload.TypeToString;
import soot.util.Chain;

public class OpenCLScene {
  private static OpenCLScene instance;
  private static int curentIdent;
  private Map<String, OpenCLClass> classes;
  private Set<OpenCLArrayType> arrayTypes;
  private MethodHierarchies methodHierarchies;
  private boolean usesGarbageCollector;
  private SootClass rootSootClass;
  private int endOfStatics;
  private Set<OpenCLInstanceof> instanceOfs;
  private List<CompositeField> compositeFields;
  private List<SootMethod> methods;
  private ClassConstantNumbers constantNumbers;
  private FieldCodeGeneration fieldCodeGeneration;
  private Map<Type, Integer> typeNumbers;
  private Set<Type> allTypes;
  private int typeNumber;
  private String cudaCode;
  
  static {
    curentIdent = 0;
  }

  public OpenCLScene(){
  }
  
  public void init(){
    classes = new LinkedHashMap<String, OpenCLClass>();
    arrayTypes = new LinkedHashSet<OpenCLArrayType>();
    methodHierarchies = new MethodHierarchies();
    instanceOfs = new HashSet<OpenCLInstanceof>();
    methods = new ArrayList<SootMethod>();
    constantNumbers = new ClassConstantNumbers();
    fieldCodeGeneration = new FieldCodeGeneration();
    typeNumbers = new HashMap<Type, Integer>();
    allTypes = new HashSet<Type>();
    loadTypes(); 
  }

  public static OpenCLScene v(){
    return instance;
  }
  
  public static void setInstance(OpenCLScene scene){
    instance = scene;
  }

  public static void releaseV(){
    instance = null;
    curentIdent++;
  }
  
  public String getIdent(){
    return "" + curentIdent;
  }

  public String getUuid(){
    return "ab850b60f96d11de8a390800200c9a66";
  }

  public int getEndOfStatics(){
    return endOfStatics;
  }

  public int getTypeNumber(Type type){
    if(typeNumbers.containsKey(type) == false){
      numberType(type);
    }
    return typeNumbers.get(type);
  }

  public int getTypeNumber(String className) {
    SootClass sootClass = Scene.v().getSootClass(className);
    return getTypeNumber(sootClass.getType());
  }
  
  public void addMethod(SootMethod sootMethod){
    SootClass sootClass = sootMethod.getDeclaringClass();
    allTypes.add(sootClass.getType());

    OpenCLClass ocl_class = getOpenCLClass(sootClass);
    ocl_class.addMethod(new OpenCLMethod(sootMethod, sootClass));

    //add the method 
    methodHierarchies.addMethod(sootMethod);
    methods.add(sootMethod);
  }
  
  public List<SootMethod> getMethods(){
    return methods;
  }

  public void addArrayType(OpenCLArrayType arrayType){
    arrayTypes.add(arrayType);
    allTypes.add(arrayType.getArrayType());
  }  
  
  public void addInstanceof(Type type){
    instanceOfs.add(new OpenCLInstanceof(type));
    allTypes.add(type);
  }

  public OpenCLClass getOpenCLClass(SootClass sootClass){    
    String className = sootClass.getName();
    if(classes.containsKey(className)){
      return classes.get(className);
    } else {
      allTypes.add(sootClass.getType());
      OpenCLClass ocl_class = new OpenCLClass(sootClass);
      classes.put(className, ocl_class);
      return ocl_class;
    }
  }

  public void addField(SootField sootField){
    SootClass sootClass = sootField.getDeclaringClass();
    OpenCLClass oclClass = getOpenCLClass(sootClass);
    oclClass.addField(new OpenCLField(sootField, sootClass));
  }

  private String getRuntimeBasicBlockClassName(){
    SootClass soot_class = rootSootClass;
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
    usesGarbageCollector = true;
  }

  public boolean getUsingGarbageCollector(){
    return usesGarbageCollector;
  }
  
  private void writeTypesToFile(){
    try {
      PrintWriter writer = new PrintWriter(RootbeerPaths.v().getTypeFile());
      for(Type type : allTypes){
        String typeStr = TypeToString.convert(type);
        writer.println(getTypeNumber(type)+" "+typeStr);
      }
      writer.flush();
      writer.close();
    } catch(Exception ex){
      ex.printStackTrace();
    }
  }
  
  public int getOutOfMemoryNumber(){
    SootClass sootClass = Scene.v().getSootClass("java.lang.OutOfMemoryError");
    return getTypeNumber(sootClass.getType());
  }
  
  private void addType(String className){
    SootClass sootClass = Scene.v().getSootClass(className);
    allTypes.add(sootClass.getType());
  }
  
  private void loadTypes(){
    addType("java.lang.OutOfMemoryError");
    addType("java.lang.StringBuilder");
    addType("java.lang.NullPointerException");
    addType("java.lang.String");
    addType("java.lang.Integer");
    addType("java.lang.Long");
    addType("java.lang.Float");
    addType("java.lang.Double");
    addType("java.lang.Boolean");
    
    Set<SootMethod> methods = DfsInfo.v().getMethods();  
    MethodSignatureUtil util = new MethodSignatureUtil();
    for(SootMethod method : methods){
      addMethod(method);
    }
    for(String extra_method : CompilerSetup.getExtraMethods()){
      util.parse(extra_method);
      addMethod(util.getSootMethod());
    }
    
    Set<SootField> fields = DfsInfo.v().getFields();
    for(SootField field : fields){
      addField(field);
    }

    FieldSignatureUtil field_util = new FieldSignatureUtil();
    ForcedFields forced_fields = new ForcedFields();
    for(String field_sig : forced_fields.get()){
      field_util.parse(field_sig);
      addField(field_util.getSootField());
    }
    
    Set<ArrayType> array_types = DfsInfo.v().getArrayTypes();
    for(ArrayType array_type : array_types){
      OpenCLArrayType ocl_array_type = new OpenCLArrayType(array_type);
      addArrayType(ocl_array_type);
    }
    for(ArrayType array_type : CompilerSetup.getExtraArrayTypes()){
      OpenCLArrayType ocl_array_type = new OpenCLArrayType(array_type);
      addArrayType(ocl_array_type);
    }
    
    Set<Type> instanceofs = DfsInfo.v().getInstanceOfs();
    for(Type type : instanceofs){
      addInstanceof(type);
    }
    
    buildCompositeFields();  
    numberTypes();
  }
  
  private void numberTypes() {
    numberType(VoidType.v());
    numberType(BooleanType.v());
    numberType(ByteType.v());
    numberType(CharType.v());
    numberType(ShortType.v());
    numberType(IntType.v());
    numberType(LongType.v());
    numberType(FloatType.v());
    numberType(DoubleType.v());
    for(Type type : allTypes){
      numberType(type);
    }
  }
  
  private void numberType(Type type){
    if(typeNumbers.containsKey(type) == false){
      typeNumbers.put(type, typeNumber);
      ++typeNumber;
    }
  }

  private String[] makeSourceCode() throws Exception {
    if(Configuration.compilerInstance().isManualCuda()){
      String filename = Configuration.compilerInstance().getManualCudaFilename();
      String cuda_code = readCode(filename);
          
      String[] ret = new String[2];
      ret[0] = cuda_code;
      ret[1] = cuda_code;
      return ret;
    }
    
    usesGarbageCollector = false;
    
    writeTypesToFile();
        
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
  
  private String readCode(String filename){
    ReadFile reader = new ReadFile(filename);
    try {
      return reader.read();
    } catch(Exception ex){
      ex.printStackTrace(System.out);
      throw new RuntimeException(ex);
    }
  }
  
  private void replaceTypeNumber(String className, String codeStr){
    SootClass sootClass = Scene.v().getSootClass(className);
    int classNumber = getTypeNumber(sootClass.getType());
    String stringClassNumber = "" + classNumber;
    cudaCode = cudaCode.replaceAll(codeStr, stringClassNumber);
  }

  private String setupEntryPoint(StringBuilder builder){
    cudaCode = builder.toString();
    String mangle = NameMangling.v().mangle(VoidType.v());
    String replacement = getRuntimeBasicBlockClassName()+"_gpuMethod"+mangle;
    //class names can have $ in them, make them regex safe
    replacement = replacement.replace("$", "\\$");
    cudaCode = cudaCode.replaceAll("%%invoke_run%%", replacement);  
    
    replaceTypeNumber("java.lang.StringBuilder", "%%java_lang_StringBuilder_TypeNumber%%");
    replaceTypeNumber("java.lang.NullPointerException", "%%java_lang_NullPointerException_TypeNumber%%");
    replaceTypeNumber("java.lang.OutOfMemoryError", "%%java_lang_OutOfMemoryError_TypeNumber%%");
    replaceTypeNumber("java.lang.String", "%%java_lang_String_TypeNumber%%");
    replaceTypeNumber("java.lang.Integer", "%%java_lang_Integer_TypeNumber%%");
    replaceTypeNumber("java.lang.Long", "%%java_lang_Long_TypeNumber%%");
    replaceTypeNumber("java.lang.Float", "%%java_lang_Float_TypeNumber%%");
    replaceTypeNumber("java.lang.Double", "%%java_lang_Double_TypeNumber%%");
    replaceTypeNumber("java.lang.Boolean", "%%java_lang_Boolean_TypeNumber%%");
    
    int size = Configuration.compilerInstance().getSharedMemSize();
    String size_str = ""+size;
    cudaCode = cudaCode.replaceAll("%%shared_mem_size%%", size_str);
    
    boolean exceptions = Configuration.compilerInstance().getExceptions();
    String exceptions_str;
    if(exceptions){
      exceptions_str = ""+1;
    } else {
      exceptions_str = ""+0;
    }
    cudaCode = cudaCode.replaceAll("%%using_exceptions%%", exceptions_str);
   
    return cudaCode;
  }
  
  public String[] getOpenCLCode() throws Exception {
    String[] source_code = makeSourceCode();
    return source_code;
  }

  public CompileResult[] getCudaCode() throws Exception {
    String[] source_code = makeSourceCode();
    return new CudaTweaks().compileProgram(source_code[0], Configuration.compilerInstance().getCompileArchitecture());
  }

  private String headerString(boolean unix) throws IOException {
    String defines = "";
    if(Configuration.compilerInstance().getArrayChecks()){
      defines += "#define ARRAY_CHECKS\n"; 
    }
    
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
    
    String barrier_path = Tweaks.v().getBarrierPath();
    String barrier_code = "";
    if(barrier_path != null){
      barrier_code = ResourceReader.getResource(barrier_path);
    }
    
    return defines + "\n" + specific_header + "\n" + both_header + "\n" + barrier_code;
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
    
    List<OpenCLMethod> methods = methodHierarchies.getMethods();
    for(OpenCLMethod method : methods){ 
      protos.add(method.getMethodPrototype());
    }    
    List<OpenCLPolymorphicMethod> poly_methods = methodHierarchies.getPolyMorphicMethods();
    for(OpenCLPolymorphicMethod poly_method : poly_methods){
      protos.add(poly_method.getMethodPrototypes());
    }
    protos.add(fieldCodeGeneration.prototypes(classes));
    for(OpenCLArrayType array_type : arrayTypes){
      protos.add(array_type.getPrototypes());
    }
    for(OpenCLInstanceof type : instanceOfs){
      protos.add(type.getPrototype());
    }
    Iterator<String> iter = protos.iterator();
    while(iter.hasNext()){
      ret.append(iter.next());
    }
    return ret.toString();
  }
  
  public List<MethodSignature> getVirtualMethods(String signature){
    List<MethodSignature> ret = new ArrayList<MethodSignature>();
    
    MethodSignatureUtil util = new MethodSignatureUtil();
    util.parse(signature);
    
    SootMethod sootMethod = util.getSootMethod();
    SootClass sootClass = sootMethod.getDeclaringClass();
    
    if(sootMethod.isStatic()){
      util.parse(sootMethod.getSignature());
      MethodSignature methodSignature = new MethodSignature(util);
      ret.add(methodSignature);
      return ret;
    }

    List<SootMethod> methodList = findVirtualHierarchy(sootClass, sootMethod);
    Collections.sort(methodList, new VirtualMethodSorter());
    
    for(SootMethod method : methodList){
      util.parse(method.getSignature());
      MethodSignature methodSignature = new MethodSignature(util);
      ret.add(methodSignature);
    }
    
    return ret;
  }
  
  private List<SootMethod> findVirtualHierarchy(SootClass sootClass, SootMethod sootMethod) {
    Set<SootMethod> returnSet = new HashSet<SootMethod>();
    if(sootMethod.isConcrete()){
      returnSet.add(sootMethod);
    }
    for(Type type : DfsInfo.v().getVirtualMethodBases()){
      if(type instanceof RefType){
        RefType refType = (RefType) type;
        SootClass virtualBaseClass = refType.getSootClass();
        MethodSignatureUtil util = new MethodSignatureUtil(sootMethod.getSignature());
        util.setClassName(virtualBaseClass.getName());
        SootMethod declaredMethod;
        try {
          declaredMethod = util.getSootMethod();
        } catch(RuntimeException ex){
          declaredMethod = null;
        }
        if(declaredMethod != null){
          if(above(declaredMethod, sootMethod)){
            returnSet.add(declaredMethod);
          } else if(above(sootMethod, declaredMethod)){
            returnSet.add(declaredMethod);
          }
        }
      }
    }
    List<SootMethod> ret = new ArrayList<SootMethod>();
    ret.addAll(returnSet);
    return ret;
  }

  private boolean above(SootMethod method1, SootMethod method2) {
    LinkedList<SootClass> queue = new LinkedList<SootClass>();
    queue.add(method2.getDeclaringClass());
    while(queue.isEmpty() == false){
      SootClass sootClass = queue.removeFirst();
      if(sootClass.getName().equals(method1.getDeclaringClass().getName())){
        return true;
      }
      if(sootClass.hasSuperclass()){
        queue.add(sootClass.getSuperclass());
      }
      for(SootClass iface : sootClass.getInterfaces()){
        queue.add(iface);
      }
    }
    return false;
  }

  private String methodBodiesString() throws IOException{
    StringBuilder ret = new StringBuilder();
    if(usesGarbageCollector){
      ret.append("#define USING_GARBAGE_COLLECTOR\n");
    }
    
    //a set is used so duplicates get filtered out
    Set<String> bodies = new HashSet<String>();
    List<OpenCLMethod> methods = methodHierarchies.getMethods();
    for(OpenCLMethod method : methods){ 
      bodies.add(method.getMethodBody());
    }
    List<OpenCLPolymorphicMethod> poly_methods = methodHierarchies.getPolyMorphicMethods();
    for(OpenCLPolymorphicMethod poly_method : poly_methods){
      bodies.addAll(poly_method.getMethodBodies());
    }
    FieldTypeSwitch type_switch = new FieldTypeSwitch();
    String field_bodies = fieldCodeGeneration.bodies(classes, type_switch);
    bodies.add(field_bodies);
    for(OpenCLArrayType array_type : arrayTypes){
      bodies.add(array_type.getBodies());
    }
    for(OpenCLInstanceof type : instanceOfs){
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
    List<CompositeField> composites = getCompositeFields();
    for(CompositeField composite : composites){
      List<SootClass> classes = composite.getClasses();
      if(classes.contains(soot_class))
        return new OffsetCalculator(composite);
    }
    throw new RuntimeException("Cannot find composite field for soot_class");
  }

  public void addCodeSegment(MethodCodeSegment codeSegment){
    rootSootClass = codeSegment.getRootSootClass();    
    getOpenCLClass(rootSootClass);
  }

  public boolean isArrayLocalWrittenTo(Local local){
    return true;
  }
  
  public boolean isRootClass(SootClass soot_class) {
    return soot_class.getName().equals(rootSootClass.getName());
  }

  public Map<String, OpenCLClass> getClassMap(){
    return classes;
  }

  public List<CompositeField> getCompositeFields() {
    return compositeFields;
  }

  private void buildCompositeFields() {
    CompositeFieldFactory factory = new CompositeFieldFactory();
    factory.setup(classes);
    compositeFields = factory.getCompositeFields();
  }
  
  public ClassConstantNumbers getClassConstantNumbers(){
    return constantNumbers;
  }

}
