/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.runtime;

import sun.awt.windows.ThemeReader;

public class ThreadConfig {

  private int m_blockShapeX;
  private int m_gridShapeX;
  
  public ThreadConfig(int block_shape_x, int grid_shape_x){
    m_blockShapeX = block_shape_x;
    m_gridShapeX = grid_shape_x;
  }
  
  public int getBlockShapeX(){
    return m_blockShapeX;
  }
  
  public int getGridShapeX(){
    return m_gridShapeX;
  }
}
