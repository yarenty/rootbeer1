/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.entry;

import java.util.ArrayList;
import java.util.List;

public class ForcedFields {
  
  private static ForcedFields m_instance = new ForcedFields();
  private List<String> m_fields;
  
  // private constructor for singleton
  private ForcedFields(){
    m_fields = new ArrayList<String>();
    m_fields.add("<java.lang.Boolean: boolean value>");
    m_fields.add("<java.lang.Integer: int value>");
    m_fields.add("<java.lang.Long: long value>");
    m_fields.add("<java.lang.Float: float value>");
    m_fields.add("<java.lang.Double: double value>");
  }
  
  public static ForcedFields getInstance() {
    return m_instance;
  }
  
  public List<String> get(){
    return m_fields;
  }
}
