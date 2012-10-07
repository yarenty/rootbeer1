/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.generate.bytecode;

import java.util.*;
import soot.ArrayType;
import soot.Type;

public class MultiDimensionalArrayTypeCreator {

  private List<ArrayType> m_arrayTypes;
  private Set<Type> m_allArrayTypes;
  private Set<Type> m_ret;

  public MultiDimensionalArrayTypeCreator(){
    m_arrayTypes = new ArrayList<ArrayType>();
    m_allArrayTypes = new HashSet<Type>();
    m_ret = new HashSet<Type>();
  }

  public Set<Type> create(Set<Type> types){
    segregateTypes(types);
    cloneArrayTypes();
    return m_ret;
  }

  private void segregateTypes(Set<Type> types) {
    for(Type type : types){
      if(type instanceof ArrayType){
        m_arrayTypes.add((ArrayType) type);
        m_allArrayTypes.add((ArrayType) type);
      } 
    }
  }

  private void cloneArrayTypes() {
    for(ArrayType type : m_arrayTypes){
      Type base_type = type.baseType;
      int dim = type.numDimensions;
      for(int i = dim - 1; i > 0; --i){
        ArrayType curr = ArrayType.v(base_type, i);
        if(m_allArrayTypes.contains(curr) == false){
          m_ret.add(curr);
          m_allArrayTypes.add(curr);
        }
      }
    }
  }
}
