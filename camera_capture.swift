// Streams both USB robot cameras at 640x480 (both fit on one USB 2.0 hub at that size; two
// 1280x720 streams don't) and keeps the latest frame of each in OUT_DIR/live_<uniqueID>.jpg,
// rewritten ~15 times a second. camera_view.py shows them in a window.
//
//   swiftc -O camera_capture.swift -o camera_capture && ./camera_capture OUT_DIR [uniqueID ...]
// With IDs, only those cameras are streamed (three cameras on one USB 2.0 hub don't all fit).
import AVFoundation
import AppKit
import CoreImage

final class Saver: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let path: URL
    var lastWrite = Date.distantPast
    init(path: URL) { self.path = path }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard Date().timeIntervalSince(lastWrite) > 0.066, let pixels = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        lastWrite = Date()
        let rep = NSBitmapImageRep(ciImage: CIImage(cvPixelBuffer: pixels))
        guard let jpeg = rep.representation(using: .jpeg, properties: [.compressionFactor: 0.8]) else { return }
        let tmp = path.appendingPathExtension("tmp")
        try? jpeg.write(to: tmp)
        _ = try? FileManager.default.replaceItemAt(path, withItemAt: tmp)  // atomic, so readers never see half a file
    }
}

let outDir = URL(fileURLWithPath: CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : ".")
let wanted = Array(CommandLine.arguments.dropFirst(2))
let found = AVCaptureDevice.DiscoverySession(deviceTypes: [.external], mediaType: .video, position: .unspecified).devices
let devices = wanted.isEmpty ? found : wanted.compactMap { id in found.first { $0.uniqueID == id } }
let preset: AVCaptureSession.Preset = ProcessInfo.processInfo.environment["CAM_SIZE"] == "320" ? .qvga320x240 : .vga640x480
var sessions: [AVCaptureSession] = []
var savers: [Saver] = []
for device in devices {
    let session = AVCaptureSession()
    session.sessionPreset = preset
    guard let input = try? AVCaptureDeviceInput(device: device), session.canAddInput(input) else { continue }
    session.addInput(input)
    let output = AVCaptureVideoDataOutput()
    let saver = Saver(path: outDir.appendingPathComponent("live_\(device.uniqueID).jpg"))
    output.setSampleBufferDelegate(saver, queue: DispatchQueue(label: device.uniqueID))
    session.addOutput(output)
    session.startRunning()
    sessions.append(session)
    savers.append(saver)
    print("streaming \(device.localizedName) \(device.uniqueID)")
}
setvbuf(stdout, nil, _IONBF, 0)
RunLoop.main.run()
