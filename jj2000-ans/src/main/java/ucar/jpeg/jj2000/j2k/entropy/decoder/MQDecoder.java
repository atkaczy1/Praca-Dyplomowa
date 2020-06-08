/*
 * CVS identifier:
 *
 * $Id: MQDecoder.java,v 1.32 2001/10/17 16:58:00 grosbois Exp $
 *
 * Class:                   MQDecoder
 *
 * Description:             Class that encodes a number of bits using the
 *                          MQ arithmetic decoder
 *
 *
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
package ucar.jpeg.jj2000.j2k.entropy.decoder;

import ucar.jpeg.jj2000.j2k.entropy.encoder.MQCoder;

/**
 * This class implements the ANS decoder.
 * */
public class MQDecoder {

    /** Length of the range in which current state of the decoder must stay.
     * Current state must stay in range of [STATE_RANGE, 2 * STATE_RANGE - 1]. */
    public static final int STATE_RANGE = 1024;

    /** Lookup table for ANS decoder. */
    final static int[] decoderLookupTable = getDecoderLookupTable();

    /** The ByteInputBuffer used to read the compressed bit stream from. */
    ByteInputBuffer in;

    /** The current state of the decoder */
    int state;

    /** Buffer to hold bits that are about to be decoded. ByteInputBuffer returns only
     * full bytes, which means we need to buffer returned bytes, so we can return
     * single bits. */
    public int bitBuffer;

    /** Number of bits that are left in bitBuffer. Value ranges between 0 and 31. */
    public int nrOfBits;

    /** The current most probable signal for each context */
    int[] mPS;

    /** The current index of each context */
    int[] I;

    /** 0xFF byte detected */
    boolean ffByteDetected;

    /**
     * Instantiates a new ANS decoder, with the specified number of contexts
     * and initial states. The compressed bytestream is read from the
     * 'iStream' object.
     *
     * @param iStream the stream that contains the coded bits
     *
     * @param nrOfContexts The number of contexts used
     *
     * @param initStates The initial state for each context. A reference is
     * kept to this array to reinitialize the contexts whenever 'reset()' or
     * 'resetCtxts()' is called.
     * */
    public MQDecoder(ByteInputBuffer iStream, int nrOfContexts, int[] initStates) {
        in = iStream;

        I = new int[nrOfContexts];
        mPS = new int[nrOfContexts];

        // Set the contexts
        resetContexts();

        // Initialize
        init();
    }

    /**
     * Decodes 'n' symbols from the bit stream using the same context
     * 'ctxt'.
     *
     * <p>Originally this method returned the decoded symbols differently if speedup
     * mode was used or not. If true was returned, then speedup mode was used
     * and the 'n' decoded symbols were all the same and it was returned in
     * bits[0] only. If false was returned then speedup mode was not used, the
     * decoded symbols were probably not all the same and they were returned in
     * bits[0], bits[1], ... bits[n-1]. Right now this method always returns false
     * and decoded bits are returned in bits[0], bits[1], ... bits[n-1]. </p>
     *
     * @param bits The array where to put the decoded symbols. Must be of
     * length 'n' or more.
     *
     * @param ctxt The context to use in decoding the symbols.
     *
     * @param n The number of symbols to decode.
     *
     * @return Always returns false.
     * */
    public final boolean fastDecodeSymbols(int[] bits, int ctxt, int n) {

        for (int j = 0; j < n; j++)
        {
            // Decode symbol
            int value = decoderLookupTable[I[ctxt] * STATE_RANGE + (state - STATE_RANGE)];
            int tempState = (value >>> 8) & 0xFFFF;
            int shift = (value >>> 24) & 0xFF;
            if (shift > 0)
                tempState = (tempState << shift) | getBits(shift);
            int symbol = (value & 0xFF) == 1 ? mPS[ctxt] : (1 - mPS[ctxt]);
            state = tempState;

            bits[j] = symbol;
        }

        return false;
    }

    /**
     * This function performs the ANS decoding. The function receives
     * an array in which to put the decoded symbols and an array of contexts
     * with which to decode them.
     *
     * <P>Each context has a current MPS and an index describing what the
     * current probability is for the LPS.
     *
     * @param bits The array where to place the decoded symbols. It should be
     * long enough to contain 'n' elements.
     *
     * @param cX The context to use in decoding each symbol.
     *
     * @param n The number of symbols to decode
     * */
    public final void decodeSymbols(int[] bits, int[] cX, int n) {
        for (int j = 0; j < n; j++)
        {
            // Decode symbol
            int value = decoderLookupTable[I[cX[j]] * STATE_RANGE + (state - STATE_RANGE)];
            int tempState = (value >>> 8) & 0xFFFF;
            int shift = (value >>> 24) & 0xFF;
            if (shift > 0)
                tempState = (tempState << shift) | getBits(shift);
            int symbol = (value & 0xFF) == 1 ? mPS[cX[j]] : (1 - mPS[cX[j]]);
            state = tempState;

            bits[j] = symbol;
        }
    }


    /**
     * Decodes one symbol from the bit stream with the given
     * context and returns its decoded value.
     *
     * <p>Each context has a current MPS and an index describing what the
     * current probability is for the LPS. </p>
     *
     * @param context The context to use in decoding the symbol
     *
     * @return The decoded symbol, 0 or 1.
     * */
    public final int decodeSymbol(int context) {

        // Decode symbol
        int value = decoderLookupTable[I[context] * STATE_RANGE + (state - STATE_RANGE)];
        int tempState = (value >>> 8) & 0xFFFF;
        int shift = (value >>> 24) & 0xFF;
        if (shift > 0)
            tempState = (tempState << shift) | getBits(shift);
        int symbol = (value & 0xFF) == 1 ? mPS[context] : (1 - mPS[context]);
        state = tempState;
        return symbol;
    }

    /**
     * Checks for past errors in the decoding process using the predictable
     * error resilient termination. After change to AND decoding this
     * method is used only for compatibility reason and always returns false.
     *
     * @return Always returns false.
     * */
    public boolean checkPredTerm() {
        return false;
    }

    /**
     * Returns the number of contexts in the arithmetic coder.
     *
     * @return The number of contexts
     **/
    public final int getNumCtxts(){
        return I.length;
    }

    /**
     * Resets a context. It reads context from input bit stream.
     *
     * @param c The number of the context (it starts at 0).
     * */
    public final void resetCtxt(int c) {
        int read = in.read();
        I[c] = read & 0x7F;
        mPS[c] = (read >> 7) & 1;
    }

    /**
     * Resets a contexts. It is called at the beginning of decoding
     * a segment. It reads contexts states with which segment was
     * encoded from input stream.
     *
     * */
    private void resetContexts() {
        for (int c = 0; c < I.length; c++)
        {
            int read = in.read();
            I[c] = read & 0x7F;
            mPS[c] = (read >> 7) & 1;
        }
    }

    /**
     * Resets a context to the original probability distribution. The original
     * probability distribution depends on the actual implementation of the
     * arithmetic coder or decoder.
     *
     * For compatibility reason this method was left, but does nothing.
     *
     * */
    public final void resetCtxts() {
        // EMPTY
    }

    /**
     * Resets the MQ decoder to start a new segment. This is like recreating a
     * new MQDecoder object with new input data.
     *
     * @param buf The byte array containing the MQ encoded data. If null the
     * current byte array is assumed.
     *
     * @param off The index of the first element in 'buf' to be decoded. If
     * negative the byte just after the previous segment is assumed, only
     * valid if 'buf' is null.
     *
     * @param len The number of bytes in 'buf' to be decoded. Any subsequent
     * bytes are taken to be 0xFF.
     * */
    public final void nextSegment(byte[] buf, int off, int len) {
        // Set the new input
        in.setByteArray(buf,off,len);
        // Reinitialize MQ
        resetContexts();
        init();
    }

    /**
     * Returns the underlying 'ByteInputBuffer' from where the MQ coded input
     * bytes are read.
     *
     * @return The underlying ByteInputBuffer.
     * */
    public ByteInputBuffer getByteInputBuffer() {
        return in;
    }

    /**
     * Initializes the state of the ANS decoder, without modifying the current
     * context states.
     *
     * <P>To have a complete reset of the MQ (as if a new MQDecoder object was
     * created) 'resetCtxts()' should be called after this method.
     * */
    private void init() {
        // --- INITDEC

        ffByteDetected = false;

        // Read data to bitBuffer
        bitBuffer = 0;
        nrOfBits = 0;
        for (int i = 0; i < 4; i++)
        {
            int readByte = in.read();
            if (readByte == -1)
                break;
            nrOfBits += 8;
            bitBuffer |= readByte << ((3 - i) * 8);
        }
        dropExtraBits();

        int rangeBitCount = 0;
        int rangeLowerBound = STATE_RANGE;
        while(rangeLowerBound > 0)
        {
            rangeBitCount++;
            rangeLowerBound >>>= 1;
        }

        // Read the initial state of the decoder
        state = (bitBuffer >> (32 - rangeBitCount)) & ((1 << rangeBitCount) - 1);
        bitBuffer <<= rangeBitCount;
        nrOfBits -= rangeBitCount;

        // End of INITDEC ---
    }

    /**
     * Returns decoder lookup tables created based on coding table created in class MQCoder
     * @return decoder table
     */
    public static int[] getDecoderLookupTable() {
        int[] decoderLookupTable = new int[47 * STATE_RANGE];
        int[] coderLookupTable = MQCoder.coderLookupTable;
        for (int i = 0; i < 47; i++)
        {
            for (int j = 0; j < STATE_RANGE; j++)
            {
                for (int k = 0; k < 2; k++)
                {
                    int coderValue = coderLookupTable[i * STATE_RANGE * 2 + j * 2 + k];
                    int nextState = ((coderValue >>> 16) & 0xFFFF) - STATE_RANGE;
                    int nrOfShifted = (coderValue >>> 8) & 0xFF;
                    int decoderIndex = i * STATE_RANGE + nextState;
                    decoderLookupTable[decoderIndex] = (k & 0xFF) | ((((j + STATE_RANGE) >> nrOfShifted) & 0xFFFF) << 8) | ((nrOfShifted & 0xFF) << 24);
                }
            }
        }

        return decoderLookupTable;
    }

    /**
     * Deletes extra 0 bits after 0xFF bytes found in the bitBuffer.
     */
    private void dropExtraBits() {
        int tempBitBuffer = 0; // bit buffer after deleting extra bits
        int tempBitBufferCount = 0; // Number of bits written to tempBitBuffer

        // Iterates over bytes in the bit buffer
        for (int i = 3; i >= 0; i--)
        {
            int offset = i * 8; // Offset of the byte in the bit buffer

            // Check if previous byte was 0xFF
            if (ffByteDetected)
            {
                // Add all the bits except extra one to the bit buffer
                int tempA = (bitBuffer << 1) & (0xFE << offset);
                tempBitBuffer |= tempA << ((24 - offset) - tempBitBufferCount);
                tempBitBufferCount += 7;
            }
            else
            {
                // Add all the bits to the bit buffer
                int tempA = bitBuffer & (0xFF << offset);
                tempBitBuffer |= tempA << ((24 - offset) - tempBitBufferCount);
                tempBitBufferCount += 8;
            }

            // Check if the current byte == 0xFF
            ffByteDetected = ((bitBuffer >>> offset) & 0xFF) == 0xFF;
        }

        bitBuffer = tempBitBuffer;
        nrOfBits = tempBitBufferCount;
    }

    /**
     * Returns bits from the input of the decoder
     * @param count - number of bits to return
     * @return returned bits (returned bits take last positions of the returned number).
     */
    private int getBits(int count) {
        int countToRead = count;
        int bits = 0;

        // Go in a loop until all wanted bits has been read
        while(countToRead > 0)
        {
            // Calculate number of bits that can be read in the current iteration
            int c = Math.min(countToRead, nrOfBits);

            int b = bitBuffer >>> (32 - c);
            bitBuffer <<= c;
            nrOfBits -= c;
            countToRead -= c;
            bits = (bits << c) | b;

            // If bit buffer is already empty, fill it
            if (nrOfBits == 0)
            {
                for (int i = 0; i < 4; i++)
                {
                    int readByte = in.read();
                    if (readByte == -1)
                        break;
                    nrOfBits += 8;
                    bitBuffer |= readByte << ((3 - i) << 3);
                }

                // Delete extra 0 bits after every 0xFF byte
                dropExtraBits();
            }
        }
        return bits;
    }
}