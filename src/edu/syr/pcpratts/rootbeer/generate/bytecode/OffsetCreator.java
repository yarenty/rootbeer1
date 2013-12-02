package edu.syr.pcpratts.rootbeer.generate.bytecode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import soot.Scene;
import soot.SootClass;
import soot.rbclassload.ClassHierarchy;
import soot.rbclassload.HierarchySootClass;
import soot.rbclassload.RootbeerClassLoader;
import soot.rbclassload.StringNumbers;

import edu.syr.pcpratts.rootbeer.generate.opencl.OpenCLClass;
import edu.syr.pcpratts.rootbeer.generate.opencl.OpenCLScene;
import edu.syr.pcpratts.rootbeer.generate.opencl.fields.OpenCLField;

public class OffsetCreator {

  public Map<String, Map<OpenCLField, Integer>> createOffsets() {
    Map<String, Map<OpenCLField, Integer>> ret = new HashMap<String, Map<OpenCLField, Integer>>();
    for(String class_name : OpenCLScene.v().getClassMap().keySet()){
      ret.put(class_name, create(class_name));
    }
    return ret;
  }

  private Map<OpenCLField, Integer> create(String class_name) {
    Map<OpenCLField, Integer> ret = new HashMap<OpenCLField, Integer>();
    
    SootClass soot_class = Scene.v().getSootClass(class_name);
    List<OpenCLField> ref_fields = collectFields(soot_class, true);
    List<OpenCLField> prim_fields = collectFields(soot_class, false);
    
    packingSort(ref_fields);
    packingSort(prim_fields);
    
    int curr_offset = Constants.SizeGcInfo;
    for(OpenCLField ref_field : ref_fields){
      ret.put(ref_field, curr_offset);
      curr_offset += ref_field.getSize();
    }
    curr_offset = align(curr_offset);
    for(OpenCLField prim_field : prim_fields){
      ret.put(prim_field, curr_offset);
      curr_offset += prim_field.getSize();
    }
    return ret;
  }

  private int align(int curr_offset) {
    int mod = curr_offset % 8;
    if(mod != 0){
      curr_offset += (8 - mod);
    }
    return curr_offset;
  }

  private void packingSort(List<OpenCLField> ref_fields) {
    Collections.sort(ref_fields, new PackingSortComparator());
  }

  private List<OpenCLField> collectFields(SootClass soot_class, boolean ref_type) {
    LinkedList<SootClass> queue = new LinkedList<SootClass>();
    List<OpenCLField> ret = new ArrayList<OpenCLField>();
    queue.add(soot_class);
    while(queue.isEmpty() == false){
      SootClass curr = queue.removeFirst();
      OpenCLClass ocl_class = OpenCLScene.v().getOpenCLClass(curr);
      
      System.out.println("collectFields: "+curr.getName());
      
      List<OpenCLField> fields;
      if(ref_type){
        fields = ocl_class.getInstanceRefFields();
      } else {
        fields = ocl_class.getInstanceNonRefFields();
      }
        
      for(OpenCLField field : fields){
        if(field.isCloned()){
          continue;
        }
        ret.add(field);
      }
            
      if(curr.hasOuterClass()){
        queue.add(curr.getOuterClass());
      }
      if(curr.hasSuperclass()){
        queue.add(curr.getSuperclass());
      }
    }
    return ret;
  }
  
  public class PackingSortComparator implements Comparator<OpenCLField> {

    @Override
    public int compare(OpenCLField lhs, OpenCLField rhs) {
      Integer this_size = lhs.getSize();
      Integer other_size = rhs.getSize();

      //sorting from highest to lowest
      int ret = other_size.compareTo(this_size);
      if(ret == 0){
        return lhs.getName().compareTo(rhs.getName());
      } else {
        return ret;
      }
    }
  }
}
