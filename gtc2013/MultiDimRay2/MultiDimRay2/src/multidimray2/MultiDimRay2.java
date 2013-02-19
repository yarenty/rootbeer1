/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package multidimray2;

import java.awt.GridLayout;
import javax.swing.JFrame;

/**
 *
 * @author thorsten
 */
public class MultiDimRay2 {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        JFrame f = new JFrame();
        f.setSize(100, 100);
        f.setLayout(new GridLayout());
        f.add(new NewJPanel());
        f.setVisible(true);
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }
}
