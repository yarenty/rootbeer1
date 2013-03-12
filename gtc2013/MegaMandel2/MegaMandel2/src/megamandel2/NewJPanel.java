/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package megamandel2;

import java.awt.Graphics;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.util.logging.Level;
import java.util.logging.Logger;
import mandellib.MandelGenerator;

import edu.syr.pcpratts.rootbeer.runtime.util.Stopwatch;

import java.util.List;
import java.util.ArrayList;

/**
 *
 * @author thorsten
 */
public class NewJPanel extends javax.swing.JPanel {

    private BufferedImage img;
    private float minx = -2;
    private float maxx = 2;
    private float miny = -2;
    private float maxy = 2;
    private static final int maxdepth = 2000;
    private float fx = 0;
    private float fy = 0;
    private float dx = 0;
    private float dy = 0;
    private boolean m_cpu;

    private static Stopwatch m_cpuWatch = new Stopwatch();

    /**
     * Creates new form NewJPanel
     */
    public NewJPanel(boolean cpu) {
        initComponents();
        img = new BufferedImage(256, 256, BufferedImage.TYPE_3BYTE_BGR);
        m_cpu = cpu;

        new Thread() {
            @Override
            public void run() {

                while (true) {
                    BufferedImage im = img;

                    int width = im.getWidth();
                    int height = im.getHeight();
                    int[] ps = new int[width*height];
  
                    if(m_cpu){
                      cpuGenerate(width, height, minx, maxx, miny, maxy, maxdepth, ps);
                    } else {
                      MandelGenerator.gpuGenerate(width, height, minx, maxx, miny, maxy, maxdepth, ps);
                    }

                    im.setRGB(0, 0, width, height, ps, 0, width);

                    float dfx = (maxx - minx) * fx;
                    float dfy = (maxy - miny) * fy;
                    maxx -= dfx;
                    minx += dfx;
                    maxy -= dfy;
                    miny += dfy;
                    maxx += dx;
                    minx += dx;
                    maxy += dy;
                    miny += dy;

                    repaint();

                    try {
                        sleep(1);
                    } catch (InterruptedException ex) {
                        Logger.getLogger(NewJPanel.class.getName()).log(Level.SEVERE, null, ex);
                    }
                }
            }
        }.start();
    }


  private void cpuGenerate(int w, int h, double minx, double maxx, double miny, double maxy, int maxdepth, int[] pixels) {
    m_cpuWatch.start();
    int num_cores = Runtime.getRuntime().availableProcessors();
    int num_each = w / num_cores;
    List<Thread> threads = new ArrayList<Thread>();
    for(int i = 0; i < num_cores; ++i){
      int start_w = i * num_each;
      int stop_w = (i + 1) * num_each;
      if(i == num_cores - 1){
        stop_w = w;
      }
      CpuThreadProc proc = new CpuThreadProc(start_w, stop_w, w, h, minx, maxx,
        miny, maxy, maxdepth, pixels);
      Thread thread = new Thread(proc);
      thread.start();
      threads.add(thread);
    }
    for(Thread thread : threads){
      try {
        thread.join();
      } catch(Exception ex){
        ex.printStackTrace();
      }
    }
    m_cpuWatch.stop();
    System.out.println("avg cpu: "+m_cpuWatch.getAverageTime());    
  }

  private class CpuThreadProc implements Runnable {

    private int m_startW;
    private int m_stopW;
    private int m_w;
    private int m_h;
    private double m_minx;
    private double m_maxx;
    private double m_miny;
    private double m_maxy;
    private int m_maxdepth;
    private int[] m_pixels;

    public CpuThreadProc(int start_w, int stop_w, int w, int h, double minx, 
      double maxx, double miny, double maxy, int maxdepth, int[] pixels){

      m_startW = start_w;
      m_stopW = stop_w;
      m_w = w;
      m_h = h;
      m_minx = minx;
      m_maxx = maxx;
      m_miny = miny;
      m_maxy = maxy;
      m_maxdepth = maxdepth;
      m_pixels = pixels;
    }

    public void run(){
      double xr = 0;
      double xi = 0;
      for(int i = m_startW; i < m_stopW; ++i){
        for(int j = 0; j < m_h; ++j){
          double cr = (m_maxx - m_minx) * i / m_w + m_minx;
          double ci = (m_maxy - m_miny) * j / m_h + m_miny;
          int d = 0;
          while (true) {
              double xr2 = xr * xr - xi * xi + cr;
              double xi2 = 2.0f * xr * xi + ci;
              xr = xr2;
              xi = xi2;
              d++;
              if (d >= m_maxdepth) {
                  break;
              }
              if (xr * xr + xi * xi >= 4) {
                  break;
              }
          }
          int r = (int) (0xff * (Math.sin((double) (0.01 * d + 0) + 1)) / 2);
          int g = (int) (0xff * (Math.sin((double) (0.02 * d + 0.01) + 1)) / 2);
          int b = (int) (0xff * (Math.sin((double) (0.04 * d + 0.1) + 1)) / 2);
          int dest_index = j * m_w + i;
   
          m_pixels[dest_index] =
                  (int) ((0xff * (0.01 * d + 0) + 1) / 2) << 16
                  | (int) ((0xff * (0.02 * d + 0.01) + 1) / 2) << 8
                  | (int) ((0xff * (0.04 * d + 0.1) + 1) / 2);
        }      
      }
    }
  }

    @Override
    protected void paintComponent(Graphics g) {
        g.drawImage(img, 0, 0, null);
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        addMouseListener(new java.awt.event.MouseAdapter() {
            public void mousePressed(java.awt.event.MouseEvent evt) {
                formMousePressed(evt);
            }
            public void mouseReleased(java.awt.event.MouseEvent evt) {
                formMouseReleased(evt);
            }
        });
        addComponentListener(new java.awt.event.ComponentAdapter() {
            public void componentResized(java.awt.event.ComponentEvent evt) {
                formComponentResized(evt);
            }
        });
        addMouseMotionListener(new java.awt.event.MouseMotionAdapter() {
            public void mouseDragged(java.awt.event.MouseEvent evt) {
                formMouseDragged(evt);
            }
        });

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(this);
        this.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 400, Short.MAX_VALUE)
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 300, Short.MAX_VALUE)
        );
    }// </editor-fold>//GEN-END:initComponents

    private void formComponentResized(java.awt.event.ComponentEvent evt) {//GEN-FIRST:event_formComponentResized
        img = new BufferedImage(getWidth(), getHeight(), BufferedImage.TYPE_3BYTE_BGR);
    }//GEN-LAST:event_formComponentResized

    private void formMousePressed(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_formMousePressed
        dx = 0.0001f * (evt.getX() - getWidth() / 2) * (maxx - minx);
        dy = 0.0001f * (evt.getY() - getHeight() / 2) * (maxy - miny);

        if (evt.getButton() == MouseEvent.BUTTON1) {
            fx = 0.01f;
            fy = 0.01f;
        } else if (evt.getButton() == MouseEvent.BUTTON3) {
            fx = -0.01f;
            fy = -0.01f;
        }
    }//GEN-LAST:event_formMousePressed

    private void formMouseReleased(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_formMouseReleased
        fx = 0;
        fy = 0;
        dx = 0;
        dy = 0;
    }//GEN-LAST:event_formMouseReleased

    private void formMouseDragged(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_formMouseDragged
        dx = 0.0001f * (evt.getX() - getWidth() / 2) * (maxx - minx);
        dy = 0.0001f * (evt.getY() - getHeight() / 2) * (maxy - miny);
    }//GEN-LAST:event_formMouseDragged
    // Variables declaration - do not modify//GEN-BEGIN:variables
    // End of variables declaration//GEN-END:variables
}
