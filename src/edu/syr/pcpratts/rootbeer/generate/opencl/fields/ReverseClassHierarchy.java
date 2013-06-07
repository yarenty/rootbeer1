/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.generate.opencl.fields;

import edu.syr.pcpratts.rootbeer.generate.opencl.OpenCLClass;
import edu.syr.pcpratts.rootbeer.generate.opencl.OpenCLScene;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import soot.Scene;
import soot.SootClass;
import soot.options.Options;
import soot.rbclassload.ClassHierarchy;
import soot.rbclassload.HierarchyGraph;
import soot.rbclassload.RootbeerClassLoader;
import soot.rbclassload.StringNumbers;

public class ReverseClassHierarchy {
  private List<TreeNode> m_Hierarchy;
  private Map<String, OpenCLClass> m_Classes;
  
  /**
   * Builds a List<TreeNode> where each is a class just below Object. The 
   * algorithm works when there are holes in the class hierarchy (a hole is
   * when the real root or a parent is not in Map<String, OpenCLClass> classes)
   * @param classes 
   */
  public ReverseClassHierarchy(Map<String, OpenCLClass> classes){
    m_Hierarchy = new ArrayList<TreeNode>();
    m_Classes = classes;
        
    Set<String> key_set = classes.keySet();
    Set<String> roots = new HashSet<String>();
    ClassHierarchy class_hierarchy = RootbeerClassLoader.v().getClassHierarchy();
    
    SootClass obj_class = Scene.v().getSootClass("java.lang.Object");
    HierarchyGraph hgraph = class_hierarchy.getHierarchyGraph(obj_class);
    
    List<Integer> queue = new LinkedList<Integer>();
    queue.add(0);                         //java.lang.Object is number 0
    
    while(!queue.isEmpty()){
      Integer class_num = queue.get(0);
      queue.remove(0);
      
      String class_name = StringNumbers.v().getString(class_num);
      if(key_set.contains(class_name) && !class_name.equals("java.lang.Object")){
        if(roots.contains(class_name)){
          continue;
        } else {
          SootClass soot_class = Scene.v().getSootClass(class_name);
          OpenCLClass ocl_class = classes.get(class_name);
          TreeNode tree = new TreeNode(soot_class, ocl_class);
          m_Hierarchy.add(tree);
          roots.add(class_name);
        }
      } else {
        queue.addAll(hgraph.getChildren(class_num));
      }
      if(roots.size() == m_Classes.size()){
        break;
      }
    }
    
    for(String class_name : m_Classes.keySet()){
      if(roots.contains(class_name)){
        continue;
      }
      List<String> up_queue = new LinkedList<String>();
      up_queue.add(class_name);
      
      while(up_queue.isEmpty() == false){
        String curr_class = up_queue.get(0);
        up_queue.remove(0);
        
        if(roots.contains(curr_class)){
          SootClass root = Scene.v().getSootClass(curr_class);
          TreeNode node = getNode(root);
          SootClass soot_class = Scene.v().getSootClass(class_name);
          OpenCLClass ocl_class = classes.get(class_name);
          node.addChild(soot_class, ocl_class);
          break;
        } else {
          int num = StringNumbers.v().addString(curr_class);
          Set<Integer> parents = hgraph.getParents(num);
          for(Integer parent : parents){
            up_queue.add(StringNumbers.v().getString(parent));
          }
        }
      }
    }
  }
  
  public List<TreeNode> get(){
    return m_Hierarchy;
  }
  
  private TreeNode getNode(SootClass soot_class){
    for(TreeNode root : m_Hierarchy){
      TreeNode ret = root.find(soot_class);
      if(ret != null)
        return ret;
    }
    return null;
  }

  private void addClass(String cls) {
    SootClass soot_class = Scene.v().getSootClass(cls);
    OpenCLClass ocl_class = OpenCLScene.v().getOpenCLClass(soot_class);
    m_Classes.put(cls, ocl_class);
  }
}
