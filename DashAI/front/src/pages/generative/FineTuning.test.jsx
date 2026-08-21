import React from "react";
import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "../../test-utils/renderWithProviders";

jest.mock("../../components/shared/FormSchemaLayout", () => ({ children }) => (
  <div>{children}</div>
));

jest.mock("../../api/datasets", () => ({
  getDatasets: jest.fn(() => Promise.resolve([{ id: 7, name: "Dolly" }])),
  getDatasetSample: jest.fn(() =>
    Promise.resolve({ instruction: ["Hi"], response: ["Hello"] }),
  ),
}));

jest.mock("../../api/fineTuning", () => ({
  getFineTuningCatalog: jest.fn(() =>
    Promise.resolve({
      models: [
        {
          key: "qwen2.5-0.5b-instruct",
          name: "Qwen2.5 0.5B Instruct",
          recommended_vram_gb: 6,
        },
      ],
      presets: {
        quick_test: {
          name: "Quick test",
          parameters: { preset: "quick_test" },
        },
      },
    }),
  ),
  getFineTuningRuns: jest.fn(() =>
    Promise.resolve([
      {
        id: 3,
        name: "Dolly adapter",
        base_model_id: "qwen2.5-0.5b-instruct",
        method: "qlora",
        dataset_id: 7,
        status: "completed",
        progress: 1,
      },
    ]),
  ),
  getLocalModels: jest.fn(() =>
    Promise.resolve([
      {
        key: "adapter:3",
        kind: "adapter",
        name: "Dolly adapter",
        source: "qwen2.5-0.5b-instruct",
        path: "E:/adapter",
        size_bytes: 1024,
      },
    ]),
  ),
  preflightFineTuning: jest.fn(),
  createFineTuningRun: jest.fn(),
  startFineTuningRun: jest.fn(),
  cancelFineTuningRun: jest.fn(),
  deleteFineTuningRun: jest.fn(),
  deleteLocalModel: jest.fn(),
}));

jest.mock("../../api/generativeTask", () => ({
  createGenerativeSession: jest.fn(),
}));

import FineTuning from "./FineTuning";
import {
  deleteLocalModel,
  getFineTuningCatalog,
  getFineTuningRuns,
  getLocalModels,
} from "../../api/fineTuning";
import { createGenerativeSession } from "../../api/generativeTask";
import { getDatasets } from "../../api/datasets";
import i18n from "../../utils/i18n";

beforeEach(() => {
  jest.clearAllMocks();
  getDatasets.mockResolvedValue([{ id: 7, name: "Dolly" }]);
  getFineTuningCatalog.mockResolvedValue({
    models: [
      {
        key: "qwen2.5-0.5b-instruct",
        name: "Qwen2.5 0.5B Instruct",
        recommended_vram_gb: 6,
      },
    ],
    presets: {
      quick_test: {
        name: "Quick test",
        parameters: { preset: "quick_test" },
      },
    },
  });
  getFineTuningRuns.mockResolvedValue([
    {
      id: 3,
      name: "Dolly adapter",
      base_model_id: "qwen2.5-0.5b-instruct",
      method: "qlora",
      dataset_id: 7,
      status: "completed",
      progress: 1,
    },
  ]);
  getLocalModels.mockResolvedValue([
    {
      key: "adapter:3",
      kind: "adapter",
      name: "Dolly adapter",
      source: "qwen2.5-0.5b-instruct",
      path: "E:/adapter",
      size_bytes: 1024,
    },
  ]);
});

describe("FineTuning page", () => {
  it("shows persisted runs and opens the four-step wizard", async () => {
    renderWithProviders(<FineTuning />, {
      route: "/app/generative/fine-tuning",
    });

    expect(await screen.findByText("Dolly adapter")).toBeInTheDocument();
    fireEvent.click(
      screen.getByRole("button", { name: /new run|nueva ejecución/i }),
    );
    await waitFor(() => {
      expect(
        screen.getByText(/model and hardware|modelo y hardware/i),
      ).toBeInTheDocument();
      expect(
        screen.getByText(/dataset mapping|mapeo del dataset/i),
      ).toBeInTheDocument();
      expect(
        screen.getByText(/configuration|configuración/i),
      ).toBeInTheDocument();
      expect(
        screen.getByText(/review and run|revisión y ejecución/i),
      ).toBeInTheDocument();
    });
  });

  it("shows the local model inventory", async () => {
    renderWithProviders(<FineTuning />, {
      route: "/app/generative/fine-tuning",
    });
    fireEvent.click(
      await screen.findByRole("tab", {
        name: /local inventory|inventario local/i,
      }),
    );
    expect(await screen.findByText("E:/adapter")).toBeInTheDocument();
  });

  it("creates an adapter-backed Generative session", async () => {
    createGenerativeSession.mockResolvedValue({ id: 11 });
    renderWithProviders(<FineTuning />, {
      route: "/app/generative/fine-tuning",
    });

    const openButton = await screen.findByRole("button", {
      name: /open in generative|abrir en generative/i,
    });
    await act(async () => fireEvent.click(openButton));

    await waitFor(() =>
      expect(createGenerativeSession).toHaveBeenCalledWith(
        expect.objectContaining({
          model_name: "PeftAdapterTextGenerationModel",
          fine_tuning_run_id: 3,
        }),
      ),
    );
  });

  it("renders through the standard i18n system in Spanish", async () => {
    await i18n.changeLanguage("es");
    try {
      renderWithProviders(<FineTuning />, {
        route: "/app/generative/fine-tuning",
      });
      expect(
        await screen.findByText("Fine-tuning local de LLM"),
      ).toBeInTheDocument();
      expect(
        screen.getByText(
          "Entrena un adaptador LoRA/QLoRA y úsalo directamente en Generative.",
        ),
      ).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: "Nueva ejecución" }),
      ).toBeInTheDocument();
    } finally {
      await i18n.changeLanguage("en");
    }
  });

  it("confirms deletion of a managed base model", async () => {
    getLocalModels.mockResolvedValue([
      {
        key: "base:qwen2.5-0.5b-instruct:main",
        kind: "base",
        name: "Qwen2.5 0.5B Instruct",
        source: "Qwen/Qwen2.5-0.5B-Instruct",
        path: "E:/model",
        size_bytes: 1024,
        in_use: false,
      },
    ]);
    jest.spyOn(window, "confirm").mockReturnValue(true);
    renderWithProviders(<FineTuning />, {
      route: "/app/generative/fine-tuning",
    });
    fireEvent.click(
      await screen.findByRole("tab", {
        name: /local inventory|inventario local/i,
      }),
    );
    const deleteButton = await screen.findByRole("button", {
      name: /delete|eliminar/i,
    });
    await act(async () => fireEvent.click(deleteButton));

    await waitFor(() =>
      expect(deleteLocalModel).toHaveBeenCalledWith(
        "base:qwen2.5-0.5b-instruct:main",
      ),
    );
    window.confirm.mockRestore();
  });
});
