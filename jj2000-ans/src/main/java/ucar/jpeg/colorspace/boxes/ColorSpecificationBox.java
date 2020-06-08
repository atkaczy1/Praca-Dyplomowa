/*****************************************************************************
 *
 * $Id: ColorSpecificationBox.java,v 1.3 2002/08/08 14:07:53 grosbois Exp $
 *
 * Copyright Eastman Kodak Company, 343 State Street, Rochester, NY 14650
 * $Date $
 *****************************************************************************/
package ucar.jpeg.colorspace.boxes;

import ucar.jpeg.jj2000.j2k.util.*;
import ucar.jpeg.jj2000.j2k.io.*;

import ucar.jpeg.colorspace.*;
import ucar.jpeg.icc.*;

import java.io.*;

/**
 * This class models the Color Specification Box in a JP2 image.
 * 
 * @version	1.0
 * @author	Bruce A. Kern
 */
public final class ColorSpecificationBox extends JP2Box {
    static { type = 0x636f6c72; }
    
    private ColorSpace.MethodEnum method     = null;
    private ColorSpace.CSEnum colorSpace = null;
    private byte[] iccProfile = null;
    private int cs, rawmethod, approxAccuracy;

    /**
     * Construct a ColorSpecificationBox from an input image.
     *   @param in RandomAccessIO jp2 image
     *   @param boxStart offset to the start of the box in the image
     * @exception IOException, ColorSpaceException 
     * */
    public ColorSpecificationBox (RandomAccessIO in, int boxStart) 
        throws IOException, ColorSpaceException {
        super(in,boxStart); 
        readBox(); 
    }

    /** Analyze the box content. */
    private void readBox() throws IOException, ColorSpaceException {
        byte[] boxHeader = new byte[256];
        in.seek (dataStart);
        in.readFully(boxHeader,0,11);
        rawmethod = boxHeader[0];
        approxAccuracy = boxHeader[2];
        switch (rawmethod) {
        case 1:    
            cs = ICCProfile.getInt(boxHeader,3);
            switch (cs) {
            case 16:
                colorSpace = ColorSpace.sRGB;
                break;  // from switch (cs)...
            case 17:
                colorSpace = ColorSpace.GreyScale;
                break;  // from switch (cs)...
            case 18:
                colorSpace = ColorSpace.sYCC;
                break;  // from switch (cs)...
            default:
		FacilityManager.getMsgLogger().
		    printmsg(MsgLogger.WARNING,
			     "Unknown enumerated colorspace (" +
			     cs + ") in color specification box");
		colorSpace = ColorSpace.Unknown;
            }
            break;  // from switch (boxHeader[0])...
        case 2:
            method = ColorSpace.ICC_PROFILED;
            cs = -1;
            int size =  ICCProfile.getInt (boxHeader, 3);
            iccProfile = new byte [size];
            in.seek(dataStart+3);
            in.readFully (iccProfile,0,size); 
            break;  // from switch (boxHeader[0])...
        default:
            throw new ColorSpaceException ("Bad specification method ("+ boxHeader[0]+") in " + this);
        }
    }

    /* Return an enumeration for the colorspace method. */
    public ColorSpace.MethodEnum getMethod () {
        return method;
    }

    /* Return an enumeration for the ucar.jpeg.colorspace. */
    public ColorSpace.CSEnum getColorSpace () {
        return colorSpace;
    }

    /**
     * Return the value of the Method field
     */
    public int getRawMethod () {
        return rawmethod;
    }

    /**
     * Return the value of the approxAccuracy field
     */
    public int getRawApproximationAccuracy () {
        return approxAccuracy;
    }

    /**
     * Return the value of the ColorSpace field
     */
    public int getRawColorSpace () {
        return cs;
    }

    /* Return a String representation of the ucar.jpeg.colorspace. */
    public String getColorSpaceString () {
        return colorSpace.value; }

    /* Return a String representation of the colorspace method. */
    public String getMethodString () {
        return method.value; }

    /* Retrieve the ICC Profile from the image as a byte []. */
    public byte [] getICCProfile () {
        return iccProfile; }

    /** Return a suitable String representation of the class instance. */
    public String toString () {
        StringBuffer rep = new StringBuffer ("[ColorSpecificationBox ");
        rep.append("method= ").append(String.valueOf(method)).append(", ");
        rep.append("ucar.jpeg.colorspace= ").append(String.valueOf(colorSpace)).
	    append("]");
        return rep.toString(); 
    }

    /* end class ColorSpecificationBox */ }









