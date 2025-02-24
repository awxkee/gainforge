/*
 * // Copyright (c) Radzivon Bartoshyk 2/2025. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
use gainforge::IsoGainMap;
use std::io;
use std::io::Read;

const ISO_METADATA_HEADER: &[u8] = b"urn:iso:std:iso:ts:21496:-1\0";
const XMP_METADATA_HEADER: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";

#[derive(Debug)]
pub(crate) struct IsoChunk {
    seq_no: u8,
    num_markers: u8,
    data: Vec<u8>,
}

pub(crate) fn find_iso_chunks<R: Read>(reader: &mut R) -> io::Result<IsoGainMap> {
    let mut buffer = [0u8; 1];
    let mut chunks = Vec::new();

    while reader.read_exact(&mut buffer).is_ok() {
        // Look for JPEG APP2 marker (0xFF E2)
        if buffer[0] == 0xFF {
            if reader.read_exact(&mut buffer).is_ok() {
                if buffer == [0xE2] {
                    // Read the segment length (big-endian)
                    let mut buffer2 = [0u8; 2];
                    reader.read_exact(&mut buffer2)?;
                    let length = u16::from_be_bytes(buffer2) as usize;

                    if length < ISO_METADATA_HEADER.len() + 2 {
                        reader.read_exact(&mut vec![0; length - 2])?; // Skip segment
                        continue;
                    }

                    // Read header
                    let mut header_buffer = vec![0u8; ISO_METADATA_HEADER.len()];
                    reader.read_exact(&mut header_buffer)?;

                    if header_buffer == ISO_METADATA_HEADER {
                        let remaining_length = length - (ISO_METADATA_HEADER.len());
                        let mut data = vec![0; remaining_length];
                        reader.read_exact(&mut data)?;

                        chunks.push(IsoChunk {
                            seq_no: 0,
                            num_markers: 0,
                            data,
                        });
                    } else {
                        reader.read_exact(&mut vec![0; length - ISO_METADATA_HEADER.len() - 2])?;
                        // Skip segment
                    }
                }
            }
        }
    }

    let mut full_chunks = vec![];
    for chunk in chunks {
        if chunk.data.len() >= size_of::<IsoGainMap>() {
            for c in chunk.data {
                full_chunks.push(c);
            }
        }
    }
    println!(
        "full_chunks: {:?}, size {}",
        full_chunks.len(),
        size_of::<IsoGainMap>()
    );

    if full_chunks.len() < size_of::<IsoGainMap>() {
        panic!("Not allowed");
    }

    let decoded = IsoGainMap::from_bytes(&full_chunks).unwrap();

    Ok(decoded)
}

pub(crate) fn find_xmp_chunk<R: Read>(reader: &mut R) -> io::Result<Vec<u8>> {
    let mut buffer = [0u8; 1];

    while reader.read_exact(&mut buffer).is_ok() {
        // Look for JPEG APP2 marker (0xFF E2)
        if buffer[0] == 0xFF {
            if reader.read_exact(&mut buffer).is_ok() {
                if buffer == [0xE1] {
                    // Read the segment length (big-endian)
                    let mut buffer2 = [0u8; 2];
                    reader.read_exact(&mut buffer2)?;
                    let length = u16::from_be_bytes(buffer2) as usize;

                    if length < XMP_METADATA_HEADER.len() + 2 {
                        reader.read_exact(&mut vec![0; length - 2])?; // Skip segment
                        continue;
                    }

                    // Read header
                    let mut header_buffer = vec![0u8; XMP_METADATA_HEADER.len()];
                    reader.read_exact(&mut header_buffer)?;

                    if header_buffer == XMP_METADATA_HEADER {
                        let remaining_length = length - (XMP_METADATA_HEADER.len());
                        let mut data = vec![0; remaining_length];
                        reader.read_exact(&mut data)?;
                        return Ok(Vec::from(data));
                    } else {
                        reader.read_exact(&mut vec![0; length - XMP_METADATA_HEADER.len() - 2])?;
                        // Skip segment
                    }
                }
            }
        }
    }

    Ok(vec![0; 0])
}
