export type ContourPoint = {
  time: number;
  value: number;
};

export type FeedbackItem = {
  code: string;
  message: string;
  severity: "info" | "warning" | "success" | string;
};

export type ToneFeedbackResponse = {
  predicted_tone: number;
  confidence: number;
  target_tone: number;
  user_contour: ContourPoint[];
  target_contour: ContourPoint[];
  five_level_value: string;
  feedback: FeedbackItem[];
};
