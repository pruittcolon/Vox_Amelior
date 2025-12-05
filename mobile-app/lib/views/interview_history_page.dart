import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:demo_ai_even/controllers/evenai_model_controller.dart';
import 'package:demo_ai_even/services/text_service.dart';
import 'package:demo_ai_even/services/evenai.dart';

class InterviewHistoryPage extends StatelessWidget {
  const InterviewHistoryPage({super.key});

  // Theme Colors (matching HomePage)
  static const Color primaryColor = Color(0xFF0D1B2A);
  static const Color cardColor = Color(0xFF1B263B);
  static const Color accentColor = Color(0xFF33A1F2);
  static const Color textColor = Colors.white;
  static const Color subtitleColor = Colors.white70;

  @override
  Widget build(BuildContext context) {
    final controller = Get.find<EvenaiModelController>();

    return Scaffold(
      backgroundColor: primaryColor,
      appBar: AppBar(
        title: const Text(
          'Interview Session',
          style: TextStyle(fontWeight: FontWeight.bold, color: textColor),
        ),
        backgroundColor: primaryColor,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new, color: textColor),
          onPressed: () {
            // When going back, we might want to stop the interview mode?
            // For now, just pop. The user can say "Terminate" to stop.
            Get.back();
          },
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.delete_sweep, color: Colors.redAccent),
            onPressed: () {
              controller.clearItems();
              Get.snackbar(
                'History Cleared',
                'All interview questions have been removed.',
                backgroundColor: cardColor,
                colorText: textColor,
                snackPosition: SnackPosition.BOTTOM,
              );
            },
          ),
        ],
      ),
      body: Column(
        children: [
          // Status Header
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(16),
            color: cardColor.withOpacity(0.5),
            child: const Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  "Live Interview Mode",
                  style: TextStyle(
                    color: accentColor,
                    fontWeight: FontWeight.bold,
                    fontSize: 16,
                  ),
                ),
                SizedBox(height: 4),
                Text(
                  "Ask questions to generate answers. Tap any card below to resend the answer to your glasses.",
                  style: TextStyle(color: subtitleColor, fontSize: 14),
                ),
              ],
            ),
          ),
          
          // Infinite Scroll List
          Expanded(
            child: Obx(() {
              if (controller.items.isEmpty) {
                return Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.mic, size: 64, color: subtitleColor.withOpacity(0.3)),
                      const SizedBox(height: 16),
                      Text(
                        "Listening for questions...",
                        style: TextStyle(color: subtitleColor.withOpacity(0.5)),
                      ),
                    ],
                  ),
                );
              }

              return ListView.separated(
                padding: const EdgeInsets.all(16),
                itemCount: controller.items.length,
                separatorBuilder: (_, __) => const SizedBox(height: 12),
                itemBuilder: (context, index) {
                  final item = controller.items[index];
                  return _InterviewItemCard(
                    question: item.title,
                    answer: item.content,
                    onTap: () async {
                      print("InterviewPage: Resending answer for '${item.title}'");
                      Get.snackbar(
                        'Sending to Glasses',
                        'Resending answer...',
                        backgroundColor: accentColor.withOpacity(0.2),
                        colorText: textColor,
                        duration: const Duration(seconds: 1),
                        snackPosition: SnackPosition.BOTTOM,
                      );
                      await TextService.get.startSendText(item.content);
                    },
                  );
                },
              );
            }),
          ),
        ],
      ),
    );
  }
}

class _InterviewItemCard extends StatelessWidget {
  const _InterviewItemCard({
    required this.question,
    required this.answer,
    required this.onTap,
  });

  final String question;
  final String answer;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Card(
      color: InterviewHistoryPage.cardColor,
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(
          color: InterviewHistoryPage.subtitleColor.withOpacity(0.1),
        ),
      ),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(16),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Question (Title)
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Icon(Icons.help_outline, color: InterviewHistoryPage.accentColor, size: 20),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      question,
                      style: const TextStyle(
                        color: InterviewHistoryPage.textColor,
                        fontWeight: FontWeight.bold,
                        fontSize: 16,
                      ),
                    ),
                  ),
                  const Icon(Icons.send_to_mobile, color: Colors.white30, size: 18),
                ],
              ),
              const SizedBox(height: 12),
              // Answer (Content Preview)
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.black26,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  answer,
                  style: const TextStyle(
                    color: InterviewHistoryPage.subtitleColor,
                    fontSize: 14,
                    height: 1.4,
                  ),
                ),
              ),
              const SizedBox(height: 8),
              const Align(
                alignment: Alignment.centerRight,
                child: Text(
                  "Tap to resend",
                  style: TextStyle(
                    color: Colors.white24,
                    fontSize: 10,
                    fontStyle: FontStyle.italic,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
