/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.generate.opencl.fields;

import edu.syr.pcpratts.rootbeer.generate.bytecode.FieldReadWriteInspector;
import edu.syr.pcpratts.rootbeer.generate.opencl.FieldPackingSorter;
import edu.syr.pcpratts.rootbeer.generate.opencl.OpenCLClass;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import soot.SootClass;

public class FieldCodeGeneration {
  
  private FieldReadWriteInspector m_Inspector;
  private FieldTypeSwitch m_TypeSwitch;
  private CompositeFieldFactory m_compositeFactory;
  
  public FieldCodeGeneration(){
    m_compositeFactory = new CompositeFieldFactory();
  }
 
  public String prototypes(Map<String, OpenCLClass> classes, FieldReadWriteInspector inspector) {
    m_Inspector = inspector;
    Set<String> set = new HashSet<String>();
    List<CompositeField> fields = m_compositeFactory.create(classes);
    for(CompositeField field : fields){
      set.addAll(getFieldPrototypes(field));
    }
    return setToString(set);
  }
  
  public String bodies(Map<String, OpenCLClass> classes, FieldReadWriteInspector inspector, FieldTypeSwitch type_switch) {
    m_Inspector = inspector;
    m_TypeSwitch = type_switch;
    Set<String> set = new HashSet<String>();
    List<CompositeField> fields = m_compositeFactory.create(classes);
    for(CompositeField field : fields){
      set.addAll(getFieldBodies(field));
    }
    return setToString(set);
  }
  
  private Set<String> getFieldBodies(CompositeField composite){
    Set<String> ret = new HashSet<String>();
    FieldPackingSorter sorter = new FieldPackingSorter();
    List<OpenCLField> ref_sorted = sorter.sort(composite.getRefFields());
    List<OpenCLField> nonref_sorted = sorter.sort(composite.getNonRefFields());
    for(OpenCLField field : ref_sorted){
      boolean writable = m_Inspector.fieldIsWrittenOnGpu(field);
      ret.add(field.getGetterSetterBodies(composite, writable, m_TypeSwitch));
    }
    for(OpenCLField field : nonref_sorted){
      boolean writable = m_Inspector.fieldIsWrittenOnGpu(field);
      ret.add(field.getGetterSetterBodies(composite, writable, m_TypeSwitch));
    }
    return ret;
  }

  private Set<String> getFieldPrototypes(CompositeField composite){
    Set<String> ret = new HashSet<String>();
    for(OpenCLField field : composite.getRefFields()){
      ret.add(field.getGetterSetterPrototypes());
    }
    for(OpenCLField field : composite.getNonRefFields()){
      ret.add(field.getGetterSetterPrototypes());
    }
    return ret;
  }
  
  private String setToString(Set<String> set){
    String ret = "";
    Iterator<String> iter = set.iterator();
    while(iter.hasNext()){
      ret += iter.next()+"\n";
    }
    return ret;
  }
}
