/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.generate.opencl.fields;

import edu.syr.pcpratts.rootbeer.generate.opencl.OpenCLClass;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import soot.SootClass;

public class CompositeFieldFactory {
  
  public List<CompositeField> create(Map<String, OpenCLClass> classes) {
    List<CompositeField> ret = new ArrayList<CompositeField>();
    for(String class_name : classes.keySet()){
      OpenCLClass ocl_class = classes.get(class_name);
      CompositeField comp = createComposite(ocl_class);
      ret.add(comp);
    }
    return ret;
  }
  
  private CompositeField createComposite(OpenCLClass ocl_class){
    SootClass soot_class = ocl_class.getSootClass();
    CompositeField composite = new CompositeField();
    for(OpenCLField field : ocl_class.getStaticRefFields()){
      composite.addRefField(field, soot_class);
    }
    for(OpenCLField field : ocl_class.getInstanceRefFields()){
      composite.addRefField(field, soot_class);
    }
    for(OpenCLField field : ocl_class.getStaticNonRefFields()){
      composite.addNonRefField(field, soot_class);
    }
    for(OpenCLField field : ocl_class.getInstanceNonRefFields()){
      composite.addNonRefField(field, soot_class);
    }
    return composite;
  }
}
