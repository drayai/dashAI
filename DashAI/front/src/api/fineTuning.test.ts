jest.mock("./api");

import api from "./api";
import {
  cancelFineTuningRun,
  createFineTuningRun,
  deleteLocalModel,
  getFineTuningRuns,
  preflightFineTuning,
  startFineTuningRun,
} from "./fineTuning";

describe("fine-tuning api", () => {
  beforeEach(() => jest.clearAllMocks());

  it("uses the public fine-tuning endpoints and preserves payloads", async () => {
    const payload = { dataset_id: 7, method: "qlora" };
    (api.post as jest.Mock).mockResolvedValue({ data: { id: 4 } });
    (api.get as jest.Mock).mockResolvedValue({ data: [{ id: 4 }] });
    (api.delete as jest.Mock).mockResolvedValue({ data: undefined });

    await preflightFineTuning(payload);
    expect(api.post).toHaveBeenCalledWith("/v1/fine-tuning/preflight", payload);
    await createFineTuningRun(payload);
    expect(api.post).toHaveBeenCalledWith("/v1/fine-tuning/runs", payload);
    await startFineTuningRun(4);
    expect(api.post).toHaveBeenCalledWith("/v1/fine-tuning/runs/4/start");
    await cancelFineTuningRun(4);
    expect(api.post).toHaveBeenCalledWith("/v1/fine-tuning/runs/4/cancel");
    await getFineTuningRuns();
    expect(api.get).toHaveBeenCalledWith("/v1/fine-tuning/runs");
    await deleteLocalModel("base:qwen:main");
    expect(api.delete).toHaveBeenCalledWith(
      "/v1/fine-tuning/models/base%3Aqwen%3Amain",
    );
  });
});
