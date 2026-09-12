import { strict as assert } from "node:assert";
import test from "node:test";
import { SidecarClient } from "../src/sidecar-client.ts";
import { FakeSidecarProcess, fakeConfig, recordRequest } from "./support.ts";

test("sidecar client maps request id to response id", async () => {
  const child = new FakeSidecarProcess();
  const client = new SidecarClient(fakeConfig(), () => child);
  recordRequest(child);
  const start = client.start();
  child.respond("handshake", {
    protocol_version: "sidecar-protocol-v0.1",
    readiness: { ready: true },
    supported_methods: ["handshake"],
  });
  await start;
  const request = client.request<{ value: number }>("begin_turn");
  child.respond("begin_turn", { turn_started: true });
  assert.deepEqual(await request, { turn_started: true });
  assert.equal(client.protocolVersion, "sidecar-protocol-v0.1");
});

test("sidecar client rejects unready handshake and stops the child", async () => {
  const child = new FakeSidecarProcess();
  const client = new SidecarClient(fakeConfig(), () => child);
  recordRequest(child);
  const start = client.start();
  child.respond("handshake", {
    protocol_version: "sidecar-protocol-v0.1",
    readiness: { ready: false, errors: ["bad corpus"] },
  });
  await assert.rejects(start, /sidecar is not ready/);
  assert.equal(client.ready, false);
  assert.ok(child.stdin.writes.includes("__end__"));
});

test("sidecar client fails closed on protocol mismatch", async () => {
  const child = new SidecarProcessProtocolMismatch();
  const client = new SidecarClient(fakeConfig(), () => child);
  recordRequest(child);
  const start = client.start();
  child.respond("handshake", {
    protocol_version: "sidecar-protocol-v0.0",
    readiness: { ready: true },
  });
  await assert.rejects(start, /protocol mismatch/);
});

class SidecarProcessProtocolMismatch extends FakeSidecarProcess {}

test("sidecar client times out pending requests", async () => {
  const child = new FakeSidecarProcess();
  const client = new SidecarClient(fakeConfig({ requestTimeoutMs: 10 }), () => child);
  recordRequest(child);
  const start = client.start();
  child.respond("handshake", {
    protocol_version: "sidecar-protocol-v0.1",
    readiness: { ready: true },
  });
  await start;
  await assert.rejects(client.request("begin_turn"), /timed out/);
});

test("sidecar client rejects pending requests on process exit", async () => {
  const child = new FakeSidecarProcess();
  const client = new SidecarClient(fakeConfig(), () => child);
  recordRequest(child);
  const start = client.start();
  child.respond("handshake", {
    protocol_version: "sidecar-protocol-v0.1",
    readiness: { ready: true },
  });
  await start;
  const request = client.request("end_turn");
  child.emitExit(1);
  await assert.rejects(request, /exited/);
});

test("sidecar client rejects invalid NDJSON and captures stderr", async () => {
  const child = new FakeSidecarProcess();
  const client = new SidecarClient(fakeConfig(), () => child);
  recordRequest(child);
  const start = client.start();
  child.respond("handshake", {
    protocol_version: "sidecar-protocol-v0.1",
    readiness: { ready: true },
  });
  await start;
  const request = client.request("end_turn");
  child.emitStdout("not-json\n");
  await assert.rejects(request, /valid NDJSON/);
  child.emitStderr("sidecar diagnostic\n");
  assert.deepEqual(client.stderr, ["sidecar diagnostic"]);
});
