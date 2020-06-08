/*****************************************************************************
 *
 * $Id: LookUpTable8Interp.java,v 1.1 2002/07/25 14:56:48 grosbois Exp $
 *
 * Copyright Eastman Kodak Company, 343 State Street, Rochester, NY 14650
 * $Date $
 *****************************************************************************/

package ucar.jpeg.icc.lut;

import ucar.jpeg.icc.tags.ICCCurveType;


/**
 * An interpolated 8 bit lut
 * 
 * @version	1.0
 * @author	Bruce A.Kern
 */
public class LookUpTable8Interp extends LookUpTable8 {

    /**
     * Construct the lut from the curve data
     *   @oaram  curve the data
     *   @oaram  dwNumInput the lut size
     *   @oaram  dwMaxOutput the lut max value
     */
    public LookUpTable8Interp (
                 ICCCurveType curve,   // Pointer to the curve data            
                 int dwNumInput,       // Number of input values in created LUT
                 byte dwMaxOutput      // Maximum output value of the LUT
                 ) {
        super (curve, dwNumInput, dwMaxOutput);

        int    dwLowIndex, dwHighIndex;		    // Indices of interpolation points
        double dfLowIndex, dfHighIndex;			// FP indices of interpolation points
        double dfTargetIndex;					// Target index into interpolation table
        double dfRatio;							// Ratio of LUT input points to curve values
        double dfLow, dfHigh;					// Interpolation values
        double dfOut;							// Output LUT value
	
        dfRatio = (double)(curve.count-1) / (double)(dwNumInput - 1);
	
        for (int i = 0; i < dwNumInput; i++) {
            dfTargetIndex = (double) i * dfRatio;
            dfLowIndex    = Math.floor(dfTargetIndex);
            dwLowIndex    = (int) dfLowIndex;
            dfHighIndex   = Math.ceil(dfTargetIndex);
            dwHighIndex   = (int) dfHighIndex;
            
            if (dwLowIndex == dwHighIndex) dfOut = ICCCurveType.CurveToDouble(curve.entry(dwLowIndex));
            else {  
                dfLow  = ICCCurveType.CurveToDouble(curve.entry(dwLowIndex));
                dfHigh = ICCCurveType.CurveToDouble(curve.entry(dwHighIndex));
                dfOut  = dfLow + (dfHigh - dfLow) * (dfTargetIndex - dfLowIndex); }
            
            lut[i] = (byte)Math.floor(dfOut * dwMaxOutput + 0.5); }}

    /* end class LookUpTable8Interp */ }
















