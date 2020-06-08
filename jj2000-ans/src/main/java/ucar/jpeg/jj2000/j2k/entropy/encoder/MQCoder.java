/*
 * CVS identifier:
 *
 * $Id: MQCoder.java,v 1.36 2002/01/10 10:31:28 grosbois Exp $
 *
 * Class:                   MQCoder
 *
 * Description:             Class that encodes a number of bits using the
 *                          MQ arithmetic coder
 *
 *
 *                          Diego SANTA CRUZ, Jul-26-1999 (improved speed)
 *
 * COPYRIGHT:
 *
 * This software module was originally developed by Raphaël Grosbois and
 * Diego Santa Cruz (Swiss Federal Institute of Technology-EPFL); Joel
 * Askelöf (Ericsson Radio Systems AB); and Bertrand Berthelot, David
 * Bouchard, Félix Henry, Gerard Mozelle and Patrice Onno (Canon Research
 * Centre France S.A) in the course of development of the JPEG2000
 * standard as specified by ISO/IEC 15444 (JPEG 2000 Standard). This
 * software module is an implementation of a part of the JPEG 2000
 * Standard. Swiss Federal Institute of Technology-EPFL, Ericsson Radio
 * Systems AB and Canon Research Centre France S.A (collectively JJ2000
 * Partners) agree not to assert against ISO/IEC and users of the JPEG
 * 2000 Standard (Users) any of their rights under the copyright, not
 * including other intellectual property rights, for this software module
 * with respect to the usage by ISO/IEC and Users of this software module
 * or modifications thereof for use in hardware or software products
 * claiming conformance to the JPEG 2000 Standard. Those intending to use
 * this software module in hardware or software products are advised that
 * their use may infringe existing patents. The original developers of
 * this software module, JJ2000 Partners and ISO/IEC assume no liability
 * for use of this software module or modifications thereof. No license
 * or right to this software module is granted for non JPEG 2000 Standard
 * conforming products. JJ2000 Partners have full right to use this
 * software module for his/her own purpose, assign or donate this
 * software module to any third party and to inhibit third parties from
 * using this software module for non JPEG 2000 Standard conforming
 * products. This copyright notice must be included in all copies or
 * derivative works of this software module.
 *
 * Copyright (c) 1999/2000 JJ2000 Partners.
 * */
package ucar.jpeg.jj2000.j2k.entropy.encoder;

import ucar.jpeg.jj2000.j2k.util.ArrayUtil;

/**
 * This class implements ANS coder. It uses tANS version of ANS.
 * Every probability model has its own coding table, from which
 * it can retrieve next state value and renormalization parameters.
 * Because coding must be done in reverse, then 3 methods used
 * to encode symbols, are used only to buffer input symbols and
 * count probabilities for bits for every context, while true
 * coding occurs in function terminate. After encoding, the output
 * buffer must be reversed again, and additional 0 bits must
 * be added after every 0xFF byte detected in the output stream.
 * Bit stuffing is done to make sure, that marks (0xFF90 - 0xFFFF)
 * don't occur in the coded stream.
 *  */
public class MQCoder {

    /** Identifier for the lazy length calculation.
     * This value is not used anymore, due to change to ANS,
     * but it was left, because it is referenced by other classes. */
    public static final int LENGTH_LAZY = 0;

    /** Identifier for a very simple length calculation.
     * This value is not used anymore, due to change to ANS,
     * but it was left, because it is referenced by other classes. */
    public static final int LENGTH_LAZY_GOOD = 1;

    /** Identifier for the near optimal length calculation.
     * This value is not used anymore, due to change to ANS,
     * but it was left, because it is referenced by other classes. */
    public static final int LENGTH_NEAR_OPT = 2;

    /** The identifier fort the termination that uses a full flush.
     * This value is not used anymore, due to change to ANS,
     * but it was left, because it is referenced by other classes. */
    public static final int TERM_FULL = 0;

    /** The identifier for the termination that uses the near optimal length
     * calculation to terminate the arithmetic codeword.
     * This value is not used anymore, due to change to ANS,
     * but it was left, because it is referenced by other classes. */
    public static final int TERM_NEAR_OPT = 1;

    /** The identifier for the easy termination that is simpler than the
     * 'TERM_NEAR_OPT' one but slightly less efficient.
     * This value is not used anymore, due to change to ANS,
     * but it was left, because it is referenced by other classes. */
    public static final int TERM_EASY = 2;

    /** The identifier for the predictable termination policy for error resilience.
     * This value is not used anymore, due to change to ANS,
     * but it was left, because it is referenced by other classes. */
    public static final int TERM_PRED_ER = 3;

    /** Length of the range in which current state of the coder must stay.
     * Current state must stay in range of [STATE_RANGE, 2 * STATE_RANGE - 1]. */
    public static final int STATE_RANGE = 1024;

    /** The data structures containing the probabilities for the LPS.
     * After change it is used to calculate probabilities of LPS and
     * write them to probabilities array. */
    final static
    int[] qe ={0x5601, 0x3401, 0x1801, 0x0ac1, 0x0521, 0x0221, 0x5601,
            0x5401, 0x4801, 0x3801, 0x3001, 0x2401, 0x1c01, 0x1601,
            0x5601, 0x5401, 0x5101, 0x4801, 0x3801, 0x3401, 0x3001,
            0x2801, 0x2401, 0x2201, 0x1c01, 0x1801, 0x1601, 0x1401,
            0x1201, 0x1101, 0x0ac1, 0x09c1, 0x08a1, 0x0521, 0x0441,
            0x02a1, 0x0221, 0x0141, 0x0111, 0x0085, 0x0049, 0x0025,
            0x0015, 0x0009, 0x0005, 0x0001, 0x5601 };

    /** Array of probabilities value for all 47 probability models used
     * in this implementation. */
    static double[] probabilities;

    /** Array of limits for ANS coder. While encoding a symbol, coder shifts bits
     * out of the state until the state is low enough that after next encoding
     * state will stay between STATE_RANGE and 2 * STATE_RANGE - 1.
     * First array dimension - probability model
     * Second array dimension - symbol that is about to be encoded */
    static int[][] limits;

    /** Lookup table for ANS coder. */
    public static int[] coderLookupTable;

    // Init coder lookup table and used probabilities and limits.
    static
    {
        probabilities = new double[47];
        limits = new int[47][2];
        getProbabilitiesAndLimits(probabilities, limits);
        coderLookupTable = getCoderLookupTable();
    }

    /** The ByteOutputBuffer used to write the final compressed bit stream. */
    ByteOutputBuffer out;

    /** The ByteOutputBuffer used to buffer compressed bit stream before reversing it
     * and writing its content to out buffer. */
    ByteOutputBuffer outBuffer;

    /** Pointer and buffer input values. We need to buffer input values
     * in order to encode input stream backwards. */
    int pointer;
    int INIT_BUFFER_SIZE = 100000;
    int BUFFER_SIZE_INCREASE = 100000;

    /** Buffer for input symbols */
    int[] symbolBuffer;

    /** Buffer for input contexts */
    int[] contextBuffer;

    /** Current state of the ANS coder */
    int state;

    /** Buffer for bits that are going to be transferred. ByteOutputBuffer accepts only
     * full bytes as input, which means we need to buffer single bits, before creating
     * full byte to write to output. */
    int bitsBuffer;

    /** The number of bits stored in bitsBuffer. Value between 0 and 31 */
    int nrOfBits;

    /** The current most probable symbol for each context */
    int[] mPS;

    /** The current index of each context */
    int[] I;

    /** The initial state of each context. Right now it is used only to remember
     * contexts that were initialized with 46 probability model. */
    int[] initStates;

    /** Array of number of bits coded in every context. Used to later calculate
     *  probabilities. */
    int[] totalCount;

    /** Array of number of 1 bits coded in every context. Used to later calculate
     *  probabilities. */
    int[] oneCount;

    /**
     * Set the length calculation type to the specified type.
     *
     * Not used anymore, it was left for compatibility reason.
     *
     * @param ltype The type of length calculation to use. One of
     * 'LENGTH_LAZY', 'LENGTH_LAZY_GOOD' or 'LENGTH_NEAR_OPT'.
     * */
    public void setLenCalcType(int ltype) {
        // EMPTY
    }

    /**
     * Set termination type to the specified type.
     *
     * Not used anymore, it was left for compatibility reason
     *
     * @param ttype The type of termination to use. One of 'TERM_FULL',
     * 'TERM_NEAR_OPT', 'TERM_EASY' or 'TERM_PRED_ER'.
     * */
    public void setTermType(int ttype) {
        // EMPTY
    }

    /**
     * Instantiates a new ANS coder, with the specified number of contexts and
     * initial states. The compressed bytestream is written to the 'oStream'
     * object.
     *
     * @param oStream where to output the compressed data.
     *
     * @param nrOfContexts The number of contexts used by the ANS coder.
     *
     * @param init The initial state for each context. A reference is kept to
     * this array to reinitialize the contexts whenever 'reset()' or
     * 'resetCtxts()' is called.
     * */
    public MQCoder(ByteOutputBuffer oStream, int nrOfContexts, int[] init) {
        out = oStream;
        outBuffer = new ByteOutputBuffer();

        I = new int[nrOfContexts];
        mPS = new int[nrOfContexts];
        initStates = init;
        state = STATE_RANGE;

        totalCount = new int[nrOfContexts];
        oneCount = new int[nrOfContexts];

        symbolBuffer = new int[INIT_BUFFER_SIZE];
        contextBuffer = new int[INIT_BUFFER_SIZE];
        pointer = 0;

        bitsBuffer = 0;
        nrOfBits = 0;

        resetCtxts();
    }

    /**
     * This method adds bit and its context n times to the buffer,
     * so it can later be encoded in terminate method.
     *
     * @param bit The symbol do code, 0 or 1.
     *
     * @param ctxt The context to us in coding the symbol.
     *
     * @param n The number of times that the symbol must be coded.
     * */
    public final void fastCodeSymbols(int bit, int ctxt, int n) {
        for (int i = 0; i < n; i++)
        {
            // Buffer symbol and context
            if (pointer >= symbolBuffer.length) {
                resizeBuffer();
            }
            symbolBuffer[pointer] = bit;
            contextBuffer[pointer] = ctxt;
            pointer++;
        }

        // Update statistics
        totalCount[ctxt] += n;
        if (bit == 1)
            oneCount[ctxt] += n;
    }

    /**
     * This method adds bits and its contexts to the buffer,
     * so it can later be encoded in terminate method.
     *
     * <p>The advantage of using this function is that the cost of the method
     * call is amortized by the number of coded symbols per method call.</p>
     *
     * @param bits An array containing the symbols to be encoded. Valid
     * symbols are 0 and 1.
     *
     * @param cX The context for each of the symbols to be encoded.
     *
     * @param n The number of symbols to encode.
     * */
    public final void codeSymbols(int[] bits, int[] cX, int n) {
        for (int i = 0; i < n; i++)
        {
            // Buffer symbol and context
            if (pointer >= symbolBuffer.length) {
                resizeBuffer();
            }
            symbolBuffer[pointer] = bits[i];
            contextBuffer[pointer] = cX[i];
            pointer++;

            // Update statistics
            totalCount[cX[i]]++;
            if (bits[i] == 1)
                oneCount[cX[i]]++;
        }
    }


    /**
     * This method adds bit and its context to the buffer,
     * so it can later be encoded in terminate method.
     *
     * @param bit The symbol to be encoded, must be 0 or 1.
     *
     * @param context the context with which to encode the symbol.
     * */
    public final void codeSymbol(int bit, int context) {

        // Buffer symbol and context
        if (pointer >= symbolBuffer.length) {
            resizeBuffer();
        }
        symbolBuffer[pointer] = bit;
        contextBuffer[pointer] = context;
        pointer++;

        // Update statistics
        totalCount[context]++;
        if (bit == 1)
            oneCount[context]++;
    }

    /**
     * This method encodes all buffered bits, then flushes remaining bits
     * and reverses output buffer, finishing encoding process.
     *
     * <p>After calling this method the 'finishLengthCalculation()' method
     * should be called, after compensating the returned length for the length
     * of previous coded segments, so that the length calculation is
     * finalized.</p>
     *
     * @return The length of the encoded codeword after termination, in
     * bytes.
     * */
    public int terminate() {

        calculateProbabilities();

        // Encoding of the all buffered symbols. Encoding is performed in reverse order
        for (int i = pointer - 1; i >= 0; i--)
        {
            int c = contextBuffer[i];
            int s = (symbolBuffer[i] == mPS[c] ? 1 : 0);

            int value = coderLookupTable[I[c] * STATE_RANGE * 2 + (state - STATE_RANGE) * 2 + s];

            int nos = (value >>> 8) & 0xFF;
            int sh = value & 0xFF;
            state = (value >>> 16) & 0xFFFF;

            // Output bits if needed
            if (nos > 0)
            {
                bitsOut(sh, nos);
            }
        }

        // Flushes state and bit buffer to output stream
        flushToOutput();

        // Add states to code stream
        for (int i = 0; i < I.length; i++)
        {
            out.write(I[i] | (mPS[i] << 7));
        }

        // Reverse bit stream and shift it to the beginning of the first byte
        byte[] reversedBuffer = getReversedBuffer();

        // Add additional 0 bit after every 0xFF byte
        return addAdditionalBits(reversedBuffer) + I.length;
    }

    /**
     * Sets probability models for every context according to collected statistics
     */
    private void calculateProbabilities() {
        for (int ctxt = 0; ctxt < I.length; ctxt++)
        {
            // It doesn't update state of the uniform context
            if (I[ctxt] != 46)
            {
                // Calculate LPS probabilities and set MPS for a context
                double p;
                if (totalCount[ctxt] == 0)
                {
                    mPS[ctxt] = 0;
                    p = 0.5;
                }
                else
                {
                    if ((oneCount[ctxt] << 1) > totalCount[ctxt])
                    {
                        mPS[ctxt] = 1;
                        p = ((double) (totalCount[ctxt] - oneCount[ctxt])) / ((double) totalCount[ctxt]);
                    }
                    else
                    {
                        mPS[ctxt] = 0;
                        p = ((double) oneCount[ctxt]) / ((double) totalCount[ctxt]);
                    }
                }

                // Find the probability model that is the closest to counted probability
                int minState = -1;
                double minDifference = 1.1;
                for (int j = 0; j < 46; j++)
                {
                    double diff = Math.abs(probabilities[j] - p);
                    if (diff < minDifference)
                    {
                        minDifference = diff;
                        minState = j;
                    }
                }

                I[ctxt] = minState;
            }
        }
    }

    /**
     * Writes bits to the output
     * @param bits - bits that are about to be placed in the output
     * @param count - number of bits placed in the output
     */
    private void bitsOut(int bits, int count) {
        int nrOfBitsToWrite = count;
        int bitsToWrite = bits;

        // Go in the loop until all bits has been written
        while(nrOfBitsToWrite > 0)
        {
            // Calculate number of bits that can be written in this iteration
            int writtenInCurrent = Math.min(32 - nrOfBits, nrOfBitsToWrite);

            // Write bits to bit buffer
            bitsBuffer >>>= writtenInCurrent;
            bitsBuffer |= bitsToWrite << (32 - writtenInCurrent);

            bitsToWrite >>>= writtenInCurrent;
            nrOfBits += writtenInCurrent;
            nrOfBitsToWrite -= writtenInCurrent;

            // If the bit buffer is full flush it to output
            if (nrOfBits == 32)
            {
                for (int i = 0; i < 4; i++)
                {
                    outBuffer.write((bitsBuffer >>> (i * 8)) & 0xFF);
                }
                nrOfBits = 0;
                bitsBuffer = 0;
            }
        }
    }

    /**
     * Returns the number of contexts in the arithmetic coder.
     *
     * @return The number of contexts
     * */
    public final int getNumCtxts() {
        return I.length;
    }

    /**
     * Resets a context to the original probability distribution, and sets its
     * more probable symbol to 0.
     *
     * @param c The number of the context (it starts at 0).
     * */
    public final void resetCtxt(int c) {
        I[c]=initStates[c];
        mPS[c] = 0;
        totalCount[c] = 0;
        oneCount[c] = 0;
    }

    /**
     * Resets all contexts to their original probability distribution and sets
     * all more probable symbols to 0.
     * */
    public final void resetCtxts() {
        System.arraycopy(initStates,0,I,0,I.length);
        ArrayUtil.intArraySet(mPS,0);
        ArrayUtil.intArraySet(totalCount,0);
        ArrayUtil.intArraySet(oneCount,0);
    }

    /**
     * Returns the number of bytes that are necessary from the compressed
     * output stream to decode all the symbols that have been coded this
     * far. The number of returned bytes does not include anything coded
     * previous to the last time the 'terminate()' or 'reset()' methods where
     * called.
     *
     * <p>The values returned by this method are then to be used in finishing
     * the length calculation with the 'finishLengthCalculation()' method,
     * after compensation of the offset in the number of bytes due to previous
     * terminated segments.</p>
     *
     * <p>This method should not be called if the current coding pass is to be
     * terminated. The 'terminate()' method should be called instead.</p>
     *
     * <p>The calculation is done based on the type of length calculation
     * specified at the constructor.</p>
     *
     * <p>Right now this function always returns 0 and later in 'finishLengthCalculation()'
     * method all the lengths are set to whole segment size.</p>
     *
     * @return The number of bytes in the compressed output stream necessary
     * to decode all the information coded this far.
     * */
    public final int getNumCodedBytes() {
        return 0;
    }

    /**
     * Reinitializes the MQ coder and the underlying 'ByteOutputBuffer' buffer
     * as if a new object was instantaited. All the data in the
     * 'ByteOutputBuffer' buffer is erased and the state and contexts of the
     * MQ coder are reinitialized). Additionally any saved MQ states are
     * discarded.
     * */
    public final void reset() {

        // Reset the output buffer
        out.reset();
        outBuffer.reset();
        pointer = 0;
        state = STATE_RANGE;

        bitsBuffer = 0;
        nrOfBits = 0;

        resetCtxts();
    }

    /**
     * Terminates the calculation of the required length for each coding
     * pass. This method must be called just after the 'terminate()' one has
     * been called for each terminated MQ segment.
     *
     * <p>The values in 'rates' must have been compensated for any offset due
     * to previous terminated segments, so that the correct index to the
     * stored coded data is used.</p>
     *
     * <p>All the lengths are set to full segment size.</p>
     *
     * @param rates The array containing the values returned by
     * 'getNumCodedBytes()' for each coding pass.
     *
     * @param n The index in the 'rates' array of the last terminated length.
     * */
    public void finishLengthCalculation(int[] rates, int n) {
        for (int i = n - 1; i >= 0; i--)
        {
            rates[i] = rates[n];
        }
    }

    /**
     * Get whole encoding lookup table. For performance coding table is stored
     * in one dimensional array.
     * @return final lookup table encapsulated in the single array.
     */
    public static int[] getCoderLookupTable() {
        int[][][][] lookupTable =  new int[47][STATE_RANGE][2][3];

        for (int i = 0; i < 46; i++)
        {
            lookupTable[i] = getCoderLookupTableForProbability(probabilities[i], limits[i]);
        }

        // Last lookup table has the same probabilities as the first one
        lookupTable[46] = lookupTable[0];

        int[] lookupTableFinal = new int[47 * STATE_RANGE * 2];
        int index = 0;
        for (int i = 0; i < 47; i++)
        {
            for (int j = 0; j < STATE_RANGE; j++)
            {
                for (int k = 0; k < 2; k++)
                {
                    lookupTableFinal[index++] = (lookupTable[i][j][k][0] & 0xFF) | ((lookupTable[i][j][k][1] & 0xFF) << 8) | ((lookupTable[i][j][k][2] & 0xFFFF) << 16);
                }
            }
        }

        return lookupTableFinal;
    }

    /**
     * Fill the arrays containing probabilities and limits used for all probabilities
     * models. For very low probabilities there are less than 2 states assigned to
     * the LPS therefore no limit value can be found. To solve this problem low
     * probabilities are risen to the last correct probability
     * @param probabilities probability array filled with calculated values
     * @param limits limits array filled with calculated values
     */
    public static void getProbabilitiesAndLimits(double[] probabilities, int[][] limits) {
        double wholeRange = qe[0] * 2;
        double last_correct_probability = 0.5;
        int[] last_correct_limits = null;
        boolean values_incorrect = false;
        for (int i = 0; i < 47; i++)
        {
            if (!values_incorrect)
            {
                double temp_prob = qe[i] / wholeRange;
                int[] temp_limits = getLimits(temp_prob);
                if (temp_limits[0] < 1)
                {
                    probabilities[i] = last_correct_probability;
                    limits[i] = last_correct_limits;
                    values_incorrect = true;
                }
                else
                {
                    probabilities[i] = last_correct_probability = temp_prob;
                    limits[i] = last_correct_limits = temp_limits;
                }
            }
            else
            {
                probabilities[i] = last_correct_probability;
                limits[i] = last_correct_limits;
            }
        }
    }

    /**
     * Get lookup table for the specified probability
     * @param p probability for which to create a table
     * @param limits limits for this probability, when following state will drop beneath this value, state is renormalized
     * @return coder table for specified probability
     */
    private static int[][][] getCoderLookupTableForProbability(double p, int[] limits) {
        int[][][] lookupTable = new int[STATE_RANGE][2][3];
        int state;
        int shiftedBits;
        int nrOfShifted;

        for (int i = 0; i < STATE_RANGE; i++)
        {
            // LPS
            state = i + STATE_RANGE;
            shiftedBits = 0;
            nrOfShifted = 0;
            while(state > limits[0])
            {
                shiftedBits += (state & 1) << nrOfShifted;
                nrOfShifted++;
                state >>= 1;
            }
            state = coderZeroEquation(state, p);
            lookupTable[i][0][0] = shiftedBits;
            lookupTable[i][0][1] = nrOfShifted;
            lookupTable[i][0][2] = state;

            // MPS
            state = i + STATE_RANGE;
            shiftedBits = 0;
            nrOfShifted = 0;
            while(state > limits[1])
            {
                shiftedBits += (state & 1) << nrOfShifted;
                nrOfShifted++;
                state >>= 1;
            }
            state = coderOneEquation(state, p);
            lookupTable[i][1][0] = shiftedBits;
            lookupTable[i][1][1] = nrOfShifted;
            lookupTable[i][1][2] = state;
        }

        return lookupTableCorrection(lookupTable);
    }

    /** Get limiting states for both LPS and MPS, so that next state will be in range
     * [STATE_RANGE, 2 * STATE_RANGE - 1]
     * @param p - probability for which calculate limits
     * @return calculated limits
     */
    private static int[] getLimits(double p) {
        int[] limits = new int[2];
        limits[0] = (int) ((2 * STATE_RANGE) * p);
        limits[1] = (int) ((2 * STATE_RANGE) * (1.0 - p));

        if (coderZeroEquation(limits[0], p) >= 2 * STATE_RANGE)
            limits[0]--;
        if (coderOneEquation(limits[1], p) >= 2 * STATE_RANGE)
            limits[1]--;

        return limits;
    }

    /**
     * Encoding equation for symbol 0
     * @param state state of the coder
     * @param probability probability with which state is encoded
     * @return next state after encoding
     */
    private static int coderZeroEquation(int state, double probability) {
        return (int)(Math.ceil(((double) (state + 1)) / probability)) - 1;
    }

    /**
     * Encoding equation for symbol 1
     * @param state state of the coder
     * @param probability probability with which state is encoded
     * @return next state after encoding
     */
    private static int coderOneEquation(int state, double probability) {
        return (int) Math.floor(((double) state) / (1.0 - probability));
    }

    /**
     * Correction to lookup tables generated through equations. Some of the state
     * transitions point to the state STATE_RANGE - 1.
     * @param lookupTable lookup table to use correction on
     * @return corrected lookup table
     */
    private static int[][][] lookupTableCorrection(int[][][] lookupTable) {
        for (int i = 0; i < STATE_RANGE; i++)
        {
            // Check if the table needs correction
            if (lookupTable[i][1][2] < STATE_RANGE)
            {
                // Change values in incorrect transition
                lookupTable[i][1][0] = 0;
                lookupTable[i][1][1] = 0;
                lookupTable[i][1][2] = 2 * STATE_RANGE - 1;

                // Change values in previous transitions
                int expectedState = 2 * STATE_RANGE - 1;
                int state = i - 1;
                while(state >= 0)
                {
                    if (lookupTable[state][1][2] == expectedState)
                    {
                        expectedState--;
                        lookupTable[state][1][2] = expectedState;
                        state--;
                    }
                    else
                    {
                        break;
                    }
                }

                // Change values in proper transitions in 0 coding table
                for (int j = 0; j < STATE_RANGE; j++)
                {
                    if (lookupTable[j][0][2] == expectedState)
                    {
                        int k;
                        for (k = j + 1; lookupTable[k][0][2] == expectedState; k++)
                        {
                            // Empty iteration
                        }
                        for (int replaceIter = j; replaceIter < k; replaceIter++)
                        {
                            lookupTable[replaceIter][0][2] = lookupTable[k][0][2];
                            lookupTable[replaceIter][0][1] = lookupTable[k][0][1];
                        }
                        break;
                    }
                }
                break;
            }
        }
        return lookupTable;
    }

    /**
     * Flushes state and bit buffer to output
     */
    private void flushToOutput() {
        int rangeBitCount = 0;
        int rangeLowerBound = STATE_RANGE;
        while(rangeLowerBound > 0)
        {
            rangeBitCount++;
            rangeLowerBound >>>= 1;
        }

        // Flush state to the bitBuffer
        bitsOut(state, rangeBitCount);

        if (nrOfBits % 8 != 0)
        {
            bitsBuffer >>>= (8 - (nrOfBits % 8));
        }

        // Flush bitBuffer to outBuffer
        int nrOfFlushed = nrOfBits / 8;
        if (nrOfBits % 8 != 0)
            nrOfFlushed++;

        while(nrOfFlushed > 0)
        {
            outBuffer.write((bitsBuffer >> ((4 - nrOfFlushed) * 8)) & 0xFF);
            nrOfFlushed--;
        }
        bitsBuffer = 0;
    }

    /**
     * Add extra 0 bits after every 0xFF byte
     * @param output array of bytes on which to operate
     * @return number of bytes added to output
     */
    private int addAdditionalBits(byte[] output) {
        int nrOfWrittenBytes = 0;

        int nrOfCarried = 0;
        int carry = 0;
        for (byte b : output) {
            int temp = carry | ((b & 0xFF) >>> nrOfCarried);
            carry = (b << (8 - nrOfCarried)) & 0xFF;
            if (temp == 0xFF) {
                nrOfCarried++;
                carry = (carry >>> 1) & 0x7F;
            }

            nrOfWrittenBytes++;
            out.write(temp);
            if (nrOfCarried == 8) {
                nrOfWrittenBytes++;
                out.write(carry);
                nrOfCarried = 0;
                carry = 0;
            }
        }

        if (nrOfCarried != 0)
        {
            nrOfWrittenBytes++;
            out.write(carry);
        }

        return nrOfWrittenBytes;
    }

    /**
     * Returns reversed byte array holding bytes written to output buffer and shifts
     * coded data so that it begins at the beginning of the first byte
     * @return reversed byte array that still needs 0 bit stuffing
     */
    private byte[] getReversedBuffer() {
        byte[] arrRev = new byte[outBuffer.size()];
        outBuffer.toByteArray(0, arrRev.length, arrRev, 0);

        int rb = nrOfBits % 8;
        int eb = 8 - rb;
        int rbMask = (1 << rb) - 1;
        int ebMask = (1 << eb) - 1;
        byte[] result = new byte[arrRev.length];
        if (rb != 0) // reverse the byte array and shift to the beginning
        {
            for (int i = arrRev.length - 1; i >= 1; i--)
            {
                int front = arrRev[i] & rbMask;
                int back = arrRev[i - 1] & (ebMask << rb);
                result[arrRev.length - 1 - i] = (byte) ((front << eb) | (back >>> rb));
            }

            result[result.length - 1] = (byte) ((arrRev[0] & rbMask) << eb);
        }
        else // no shift is required
        {
            for (int i = arrRev.length - 1; i >= 0; i--)
            {
                result[arrRev.length - 1 - i] = arrRev[i];
            }
        }

        return result;
    }

    /**
     * Resizes buffers for input data if needed
     */
    private void resizeBuffer() {
        int[] newSymbolBuffer = new int[symbolBuffer.length + BUFFER_SIZE_INCREASE];
        int[] newContextBuffer = new int[symbolBuffer.length + BUFFER_SIZE_INCREASE];
        System.arraycopy(symbolBuffer,0, newSymbolBuffer,0, symbolBuffer.length);
        System.arraycopy(contextBuffer,0, newContextBuffer,0, contextBuffer.length);
        symbolBuffer = newSymbolBuffer;
        contextBuffer = newContextBuffer;
    }
}
